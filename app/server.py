import asyncio
import contextlib
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


OPENAI_BASE_URL = env("OPENAI_BASE_URL", "http://vllm:8000/v1").rstrip("/")
SERVED_MODEL_NAME = env("SERVED_MODEL_NAME", "qwen3-asr-corrector")
REQUEST_TIMEOUT_SECONDS = float(env("REQUEST_TIMEOUT_SECONDS", "180"))
MODEL_CONTEXT_TOKENS = int(env("CORRECTION_MODEL_CONTEXT_TOKENS", env("VLLM_MAX_MODEL_LEN", "8192")))
CORRECTION_TOKEN_MARGIN = int(env("CORRECTION_TOKEN_MARGIN", "256"))
CORRECTION_OVERLAP_TOKENS = int(env("CORRECTION_OVERLAP_TOKENS", "384"))
MIN_COMPLETION_TOKENS = int(env("CORRECTION_MIN_COMPLETION_TOKENS", "256"))
FALLBACK_TOKENS_PER_WORD = float(env("CORRECTION_FALLBACK_TOKENS_PER_WORD", "1.25"))
HEALTHCHECK_PROMPT = "/no_think\nReply with READY only."
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    word: str = Field(min_length=1)


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    start: float
    end: float
    text: str = ""
    words: list[Word] = Field(min_length=1)


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[Segment] = Field(min_length=1)
    glossary: dict[str, str] = Field(default_factory=dict)
    recording_context: str = ""


class WordAnchor(BaseModel):
    segment_id: int
    word_index: int


class Correction(BaseModel):
    op: Literal["replace", "delete", "insert"]
    start: WordAnchor
    end: WordAnchor
    before: list[str]
    after: list[str]


class CorrectionResponse(BaseModel):
    corrections: list[Correction]


@dataclass(frozen=True)
class FlatWord:
    segment_id: int
    word_index: int
    word: str


@dataclass(frozen=True)
class SpanUnit:
    flat_start: int
    flat_end: int
    estimated_tokens: int


@dataclass(frozen=True)
class CorrectionWindow:
    unit_start: int
    unit_end: int
    flat_start: int
    flat_end: int
    primary_start: int
    primary_end: int
    transcript_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class WordEdit:
    start: int
    end: int
    before: list[str]
    after: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
    app.state.ready = False
    app.state.warmup_error = None
    app.state.warmup_task = asyncio.create_task(_warmup_loop(app))
    try:
        yield
    finally:
        app.state.warmup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.warmup_task
        await app.state.client.aclose()


app = FastAPI(title="Qwen ASR Corrector", lifespan=lifespan)


def _schedule_warmup(app: FastAPI) -> None:
    task = getattr(app.state, "warmup_task", None)
    if task is None or task.done():
        app.state.warmup_task = asyncio.create_task(_warmup_loop(app))


async def _warmup_loop(app: FastAPI) -> None:
    while True:
        try:
            await _probe_backend(app, perform_generation=True)
            app.state.ready = True
            app.state.warmup_error = None
            return
        except Exception as exc:  # noqa: BLE001
            app.state.ready = False
            app.state.warmup_error = str(exc)
            await asyncio.sleep(5)


async def _probe_backend(app: FastAPI, perform_generation: bool) -> None:
    models_response = await app.state.client.get(f"{OPENAI_BASE_URL}/models")
    models_response.raise_for_status()
    if not perform_generation:
        return

    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Return only the final answer."},
            {"role": "user", "content": HEALTHCHECK_PROMPT},
        ],
        "temperature": 0,
        "max_tokens": 8,
    }
    response = await app.state.client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload)
    response.raise_for_status()
    choices = response.json().get("choices", [])
    if not choices:
        raise RuntimeError("Warmup completion returned no choices")


def _ensure_ready(app: FastAPI) -> None:
    if app.state.ready:
        return
    _schedule_warmup(app)
    raise HTTPException(status_code=503, detail="Model is still warming up")


def _normalize_glossary(glossary: dict[str, str]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for term, cue in glossary.items():
        cleaned_term = term.strip()
        cleaned_cue = cue.strip()
        if not cleaned_term or cleaned_term in seen:
            continue
        normalized.append((cleaned_term, cleaned_cue))
        seen.add(cleaned_term)
    return normalized


def _format_glossary(glossary: list[tuple[str, str]]) -> str:
    lines = [f"- {term}: {cue}" if cue else f"- {term}" for term, cue in glossary]
    return "Glossary terms:\n" + "\n".join(lines)


def _flatten_words(segments: list[Segment]) -> list[FlatWord]:
    flat_words: list[FlatWord] = []
    for segment in segments:
        for word_index, word in enumerate(segment.words):
            normalized = word.word.strip()
            if normalized:
                flat_words.append(
                    FlatWord(
                        segment_id=segment.id,
                        word_index=word_index,
                        word=normalized,
                    )
                )
    return flat_words


def _transcript_from_words(words: list[str], start: int, end: int) -> str:
    return " ".join(words[start:end])


def _build_prompt(
    transcript: str,
    glossary: list[tuple[str, str]],
    recording_context: str,
) -> list[dict[str, str]]:
    instruction_lines = [
        "Correct this ASR transcript only for misspellings of glossary terms.",
        "Only change words or phrases that clearly refer to a glossary term.",
        "Use glossary spellings exactly, including capitalization.",
        "Do not add punctuation, rewrite generic words, or make grammar or style edits.",
        "If a phrase is not clearly a glossary term, leave it unchanged.",
        "Return only the corrected transcript text.",
    ]

    user_prompt = "\n\n".join(
        [
            "/no_think",
            "\n".join(instruction_lines),
            f"Recording context:\n{recording_context.strip()}" if recording_context.strip() else "Recording context:\nNone",
            _format_glossary(glossary) if glossary else "Glossary terms:\n- None",
            f"Transcript:\n{transcript}",
        ]
    )
    return [
        {"role": "system", "content": "You are a careful ASR transcript correction assistant."},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_model_output(content: str) -> str:
    normalized = THINK_BLOCK_RE.sub("", content).strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        normalized = normalized.strip("`").strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        try:
            decoded = json.loads(normalized)
        except json.JSONDecodeError:
            return normalized
        if isinstance(decoded, str):
            return decoded.strip()
    return normalized


def _parse_token_count(payload: dict[str, Any]) -> int | None:
    for key in ("token_count", "count", "input_tokens", "num_tokens", "total_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    token_ids = payload.get("token_ids")
    if isinstance(token_ids, list):
        return len(token_ids)
    return None


async def _count_text_tokens(app: FastAPI, text: str) -> int:
    if not text.strip():
        return 0

    for payload in (
        {"model": SERVED_MODEL_NAME, "prompt": text},
        {"model": SERVED_MODEL_NAME, "text": text},
    ):
        try:
            response = await app.state.client.post(f"{OPENAI_BASE_URL.removesuffix('/v1')}/tokenize", json=payload)
            if response.is_success:
                count = _parse_token_count(response.json())
                if count is not None:
                    return count
        except Exception:  # noqa: BLE001
            continue

    return max(1, int(len(text.split()) * FALLBACK_TOKENS_PER_WORD))


async def _count_message_tokens(app: FastAPI, messages: list[dict[str, str]]) -> int:
    try:
        response = await app.state.client.post(
            f"{OPENAI_BASE_URL}/messages/count_tokens",
            json={"model": SERVED_MODEL_NAME, "messages": messages},
        )
        if response.is_success:
            count = _parse_token_count(response.json())
            if count is not None:
                return count
    except Exception:  # noqa: BLE001
        pass

    raw_text = "\n".join(message["content"] for message in messages)
    return await _count_text_tokens(app, raw_text) + 32


async def _call_model(
    app: FastAPI,
    transcript: str,
    glossary: list[tuple[str, str]],
    recording_context: str,
    max_tokens: int,
) -> str:
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": _build_prompt(transcript, glossary, recording_context),
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    response = await app.state.client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _normalize_model_output(content)


def _anchor_from_global(flat_words: list[FlatWord], flat_boundary: int) -> WordAnchor:
    if not flat_words:
        return WordAnchor(segment_id=0, word_index=0)
    if flat_boundary >= len(flat_words):
        last = flat_words[-1]
        return WordAnchor(segment_id=last.segment_id, word_index=last.word_index + 1)
    word = flat_words[flat_boundary]
    return WordAnchor(segment_id=word.segment_id, word_index=word.word_index)


def _owns_correction(start_word: int, primary_start: int, primary_end: int, total_words: int) -> bool:
    if primary_start <= start_word < primary_end:
        return True
    return start_word == total_words and primary_end == total_words


def _build_word_edits(
    source_words: list[str],
    base_start: int,
    primary_start: int,
    primary_end: int,
    total_words: int,
    corrected_words: list[str],
) -> list[WordEdit]:
    matcher = SequenceMatcher(a=source_words, b=corrected_words)
    edits: list[WordEdit] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        global_i1 = base_start + i1
        global_i2 = base_start + i2
        if not _owns_correction(global_i1, primary_start, primary_end, total_words):
            continue
        edits.append(
            WordEdit(
                start=global_i1,
                end=global_i2,
                before=source_words[i1:i2],
                after=corrected_words[j1:j2],
            )
        )
    return edits


def _apply_word_edits(words: list[str], edits: list[WordEdit]) -> list[str]:
    updated = words[:]
    for edit in sorted(edits, key=lambda item: (item.start, item.end), reverse=True):
        updated[edit.start:edit.end] = edit.after
    return updated


def _build_corrections(flat_words: list[FlatWord], corrected_words: list[str]) -> list[Correction]:
    source_words = [word.word for word in flat_words]
    matcher = SequenceMatcher(a=source_words, b=corrected_words)
    corrections: list[Correction] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        corrections.append(
            Correction(
                op=op,
                start=_anchor_from_global(flat_words, i1),
                end=_anchor_from_global(flat_words, i2),
                before=source_words[i1:i2],
                after=corrected_words[j1:j2],
            )
        )
    return corrections


def _build_units(words: list[str], max_unit_tokens: int) -> list[SpanUnit]:
    units: list[SpanUnit] = []
    if not words:
        return units

    unit_size = max(8, min(32, max(1, max_unit_tokens // 8)))
    start = 0
    while start < len(words):
        end = min(start + unit_size, len(words))
        units.append(
            SpanUnit(
                flat_start=start,
                flat_end=end,
                estimated_tokens=max(
                    1,
                    int((end - start) * FALLBACK_TOKENS_PER_WORD),
                ),
            )
        )
        start = end
    return units


def _build_estimated_window(units: list[SpanUnit], start_unit: int, target_tokens: int) -> int:
    end_unit = start_unit
    total_tokens = 0
    while end_unit < len(units):
        next_cost = units[end_unit].estimated_tokens + (1 if end_unit > start_unit else 0)
        if end_unit > start_unit and total_tokens + next_cost > target_tokens:
            break
        total_tokens += next_cost
        end_unit += 1
        if total_tokens >= target_tokens:
            break
    return max(end_unit, start_unit + 1)


async def _fit_window(
    app: FastAPI,
    words: list[str],
    units: list[SpanUnit],
    start_unit: int,
    end_unit: int,
    glossary: list[tuple[str, str]],
    recording_context: str,
) -> tuple[int, int, int]:
    while True:
        flat_start = units[start_unit].flat_start
        flat_end = units[end_unit - 1].flat_end
        transcript = _transcript_from_words(words, flat_start, flat_end)
        transcript_tokens = await _count_text_tokens(app, transcript)
        input_tokens = await _count_message_tokens(app, _build_prompt(transcript, glossary, recording_context))
        completion_budget = MODEL_CONTEXT_TOKENS - input_tokens - CORRECTION_TOKEN_MARGIN
        if completion_budget >= max(MIN_COMPLETION_TOKENS, transcript_tokens):
            return end_unit, transcript_tokens, completion_budget
        if end_unit - start_unit <= 1:
            raise HTTPException(
                status_code=413,
                detail="A single transcript span is too large for the available model context",
            )
        end_unit -= 1


def _next_window_start(units: list[SpanUnit], start_unit: int, end_unit: int, overlap_tokens: int) -> int:
    next_start = end_unit
    overlap = 0
    while next_start > start_unit + 1 and overlap < overlap_tokens:
        next_start -= 1
        overlap += units[next_start].estimated_tokens
    return next_start if next_start > start_unit else end_unit


async def _build_windows(
    app: FastAPI,
    words: list[str],
    glossary: list[tuple[str, str]],
    recording_context: str,
) -> list[CorrectionWindow]:
    prompt_tokens = await _count_message_tokens(app, _build_prompt("", glossary, recording_context))
    usable_tokens = MODEL_CONTEXT_TOKENS - prompt_tokens - CORRECTION_TOKEN_MARGIN
    if usable_tokens < MIN_COMPLETION_TOKENS * 2:
        raise HTTPException(status_code=500, detail="Configured context window is too small")

    target_transcript_tokens = max(MIN_COMPLETION_TOKENS, usable_tokens // 2)
    overlap_tokens = min(CORRECTION_OVERLAP_TOKENS, max(64, target_transcript_tokens // 4))
    units = _build_units(words, target_transcript_tokens)
    windows: list[CorrectionWindow] = []

    start_unit = 0
    while start_unit < len(units):
        estimated_end = _build_estimated_window(units, start_unit, target_transcript_tokens)
        end_unit, transcript_tokens, completion_tokens = await _fit_window(
            app,
            words,
            units,
            start_unit,
            estimated_end,
            glossary,
            recording_context,
        )
        if end_unit >= len(units):
            next_start = len(units)
            primary_end = units[-1].flat_end
        else:
            next_start = _next_window_start(units, start_unit, end_unit, overlap_tokens)
            primary_end_unit = next_start if next_start > start_unit else end_unit
            primary_end = units[primary_end_unit - 1].flat_end

        windows.append(
            CorrectionWindow(
                unit_start=start_unit,
                unit_end=end_unit,
                flat_start=units[start_unit].flat_start,
                flat_end=units[end_unit - 1].flat_end,
                primary_start=units[start_unit].flat_start,
                primary_end=primary_end,
                transcript_tokens=transcript_tokens,
                completion_tokens=max(MIN_COMPLETION_TOKENS, completion_tokens),
            )
        )
        start_unit = next_start

    return windows


async def _build_glossary_batches(
    app: FastAPI,
    glossary: dict[str, str],
    recording_context: str,
) -> list[list[tuple[str, str]]]:
    normalized = _normalize_glossary(glossary)
    if not normalized:
        return []

    empty_prompt_tokens = await _count_message_tokens(app, _build_prompt("", [], recording_context))
    max_prompt_tokens = MODEL_CONTEXT_TOKENS - CORRECTION_TOKEN_MARGIN - (2 * MIN_COMPLETION_TOKENS)
    if empty_prompt_tokens >= max_prompt_tokens:
        raise HTTPException(status_code=500, detail="Configured context window is too small")

    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    for item in normalized:
        trial = current + [item]
        trial_prompt_tokens = await _count_message_tokens(app, _build_prompt("", trial, recording_context))
        if trial_prompt_tokens > max_prompt_tokens and current:
            batches.append(current)
            current = [item]
            continue
        current = trial

    if current:
        batches.append(current)

    return batches


async def _run_glossary_pass(
    app: FastAPI,
    words: list[str],
    glossary: list[tuple[str, str]],
    recording_context: str,
) -> list[str]:
    if not words or not glossary:
        return words

    windows = await _build_windows(app, words, glossary, recording_context)
    edits: list[WordEdit] = []

    for window in windows:
        transcript = _transcript_from_words(words, window.flat_start, window.flat_end)
        corrected_text = await _call_model(
            app,
            transcript,
            glossary,
            recording_context,
            max_tokens=min(window.completion_tokens, window.transcript_tokens + 128),
        )
        edits.extend(
            _build_word_edits(
                words[window.flat_start:window.flat_end],
                window.flat_start,
                window.primary_start,
                window.primary_end,
                len(words),
                corrected_text.split(),
            )
        )

    return _apply_word_edits(words, edits)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def ready() -> JSONResponse:
    if not app.state.ready:
        _schedule_warmup(app)
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "detail": app.state.warmup_error},
        )

    try:
        await _probe_backend(app, perform_generation=False)
    except Exception as exc:  # noqa: BLE001
        app.state.ready = False
        app.state.warmup_error = str(exc)
        _schedule_warmup(app)
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": str(exc)})

    return JSONResponse(status_code=200, content={"status": "ready", "model": SERVED_MODEL_NAME})


@app.post("/correct", response_model=CorrectionResponse)
async def correct(request: CorrectionRequest) -> CorrectionResponse:
    _ensure_ready(app)
    flat_words = _flatten_words(request.segments)
    if not flat_words:
        return CorrectionResponse(corrections=[])
    glossary_batches = await _build_glossary_batches(app, request.glossary, request.recording_context)
    if not glossary_batches:
        return CorrectionResponse(corrections=[])

    corrected_words = [word.word for word in flat_words]
    for glossary_batch in glossary_batches:
        corrected_words = await _run_glossary_pass(
            app,
            corrected_words,
            glossary_batch,
            request.recording_context,
        )

    return CorrectionResponse(corrections=_build_corrections(flat_words, corrected_words))

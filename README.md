# Qwen Segment Correction Stack

Minimal Docker Compose stack that exposes an HTTP service around `Qwen/Qwen3-14B-AWQ` for correcting domain-specific term misspellings in pre-segmented ASR output.

## What it does

- Serves `POST /correct` for segment-based ASR correction.
- Accepts one JSON body containing `segments`, a simple `glossary` that maps each preferred term to a brief definition or cue, and an optional `recording_context` describing the device, placement, or setting.
- Returns only structured corrections, anchored back to the original segment/word positions.
- Uses token-budgeted overlapping windows for long transcripts, and if the glossary itself is too large it automatically splits the glossary into multiple correction passes.
- Marks the public API healthy only after the Qwen model is fully loaded and a generation probe succeeds.

## Files

- `compose.yaml`: stack definition for `vllm` and the public `corrector` API.
- `app/server.py`: FastAPI wrapper, glossary-only prompting logic, segment diff mapping, and readiness checks.
- `test.segments.json`: sample segmented input fixture used by the smoke test.
- `scripts/smoke-test.sh`: builds the stack, waits for readiness, submits the fixture, verifies corrections are returned, and shuts the stack down cleanly.
- `.env`: runtime settings and secrets.
- `.env.example`: starter template if you need to recreate `.env`.

Model weights are cached in the repo-local `./.cache/models/huggingface` directory.

Key runtime knobs in `.env`:

- `VLLM_QUANTIZATION=awq_marlin`
- `VLLM_GPU_MEMORY_UTILIZATION=0.88`
- `VLLM_CPU_OFFLOAD_GB=0`
- `VLLM_MAX_MODEL_LEN=8192`
- `VLLM_MAX_NUM_SEQS=4`
- `CORRECTION_MODEL_CONTEXT_TOKENS=8192`
- `CORRECTION_OVERLAP_TOKENS=384`

On the local `NVIDIA RTX PRO 4000 Blackwell 24 GB`, this profile ran fully in VRAM:

- Model weights: about `9.37 GiB`
- KV cache: about `9.62 GiB`
- Total cached tokens: about `63,056`
- Reported concurrency at `8,192` tokens/request: about `7.7x`

## API

`POST /correct`

```json
{
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.2,
      "text": "please send this to evergreen terris",
      "words": [
        {"start": 0.0, "end": 0.2, "word": "please"},
        {"start": 0.2, "end": 0.4, "word": "send"}
      ]
    }
  ],
  "recording_context": "Ceiling-mounted kitchen microphone capturing a family conversation with room echo and appliance noise.",
  "glossary": {
    "Evergreen Terrace": "A street name and place reference.",
    "ASR Ops": "The speech operations team and related notes.",
    "Jon Smythe": "A person's full name."
  }
}
```

Example response:

```json
{
  "corrections": [
    {
      "op": "replace",
      "start": {"segment_id": 0, "word_index": 4},
      "end": {"segment_id": 0, "word_index": 6},
      "before": ["evergreen", "terris"],
      "after": ["Evergreen", "Terrace"]
    }
  ]
}
```

`start` is inclusive and `end` is exclusive. Applying the returned operations to the original `segments[].words[].word` values is enough to derive corrected text.

If `glossary` is empty, the service returns no corrections.
`recording_context` is optional and should stay brief.

## Long transcripts

The service does not assume the whole transcript or glossary fits in one prompt. It:

- estimates and then verifies token usage against the live model tokenizer when available,
- slices the flattened transcript into overlapping windows that fit within the configured correction budget,
- splits oversized glossaries into multiple focused passes so each pass stays within budget,
- reserves output space before each model call instead of filling the entire context with input,
- assigns ownership of each overlap to exactly one window so merged corrections stay precise and non-duplicated.

By default `CORRECTION_MODEL_CONTEXT_TOKENS` matches the intended serving context. If you change `VLLM_MAX_MODEL_LEN`, keep `CORRECTION_MODEL_CONTEXT_TOKENS` aligned unless you intentionally want the API to use a smaller safety budget.

## Run

1. Copy `.env.example` to `.env` if needed and set `HF_TOKEN`.
   The default runtime settings are intentionally conservative so the 14B AWQ model can start on constrained 24 GB GPUs that do not have all VRAM free.
2. Start the stack:

```bash
docker compose up -d --build
```

3. Check readiness:

```bash
docker compose ps
curl http://127.0.0.1:8080/health/ready
```

4. Stop it:

```bash
docker compose down --remove-orphans
```

## Smoke test

```bash
./scripts/smoke-test.sh
```

The smoke test waits until the public service is healthy, submits `test.segments.json`, then submits a deliberately long repeated payload under a reduced correction budget to force transcript windowing, then submits a glossary-heavy payload to force glossary batching, verifies structured corrections are returned in all cases, and then tears the stack down.

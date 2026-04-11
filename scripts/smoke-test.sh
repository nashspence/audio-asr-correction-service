#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/compose.yaml"
ENV_FILE="${ROOT_DIR}/.env"
TEST_SEGMENTS="${ROOT_DIR}/test.segments.json"
STACK_UP=0
SMOKE_CORRECTION_MODEL_CONTEXT_TOKENS="${SMOKE_CORRECTION_MODEL_CONTEXT_TOKENS:-1024}"

cleanup() {
  if [[ "${STACK_UP}" -eq 1 ]]; then
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" down --remove-orphans
  fi
}

trap cleanup EXIT

if [[ ! -f "${ENV_FILE}" ]]; then
  echo ".env is required"
  exit 1
fi

if [[ ! -f "${TEST_SEGMENTS}" ]]; then
  echo "Missing ${TEST_SEGMENTS}"
  exit 1
fi

CORRECTION_MODEL_CONTEXT_TOKENS="${SMOKE_CORRECTION_MODEL_CONTEXT_TOKENS}" \
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --build
STACK_UP=1

container_id="$(CORRECTION_MODEL_CONTEXT_TOKENS="${SMOKE_CORRECTION_MODEL_CONTEXT_TOKENS}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" ps -q corrector)"
if [[ -z "${container_id}" ]]; then
  echo "Could not find corrector container"
  exit 1
fi

for attempt in $(seq 1 80); do
  health_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health_status}" == "healthy" ]]; then
    break
  fi
  if [[ "${health_status}" == "unhealthy" ]]; then
    CORRECTION_MODEL_CONTEXT_TOKENS="${SMOKE_CORRECTION_MODEL_CONTEXT_TOKENS}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" logs --tail=200
    echo "Stack became unhealthy before ready"
    exit 1
  fi
  sleep 15
done

health_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
if [[ "${health_status}" != "healthy" ]]; then
  CORRECTION_MODEL_CONTEXT_TOKENS="${SMOKE_CORRECTION_MODEL_CONTEXT_TOKENS}" docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" logs --tail=200
  echo "Stack did not become healthy in time"
  exit 1
fi

response_file="$(mktemp)"
docker cp "${TEST_SEGMENTS}" "${container_id}:/tmp/test.segments.json"

docker exec "${container_id}" sh -lc "python - <<'PY' > /tmp/smoke-response.json
import json
import httpx

with open('/tmp/test.segments.json', 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

response = httpx.post(
    'http://127.0.0.1:8080/correct',
    json=payload,
    timeout=300,
)

print(response.status_code)
print(response.text)
PY"

docker cp "${container_id}:/tmp/smoke-response.json" "${response_file}"

python3 - <<'PY' "${response_file}"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    lines = handle.read().splitlines()

if not lines:
    raise SystemExit("Smoke response was empty")

status_code = lines[0].strip()
if status_code != "200":
    raise SystemExit(f"Smoke request failed with HTTP {status_code}")

payload = json.loads("\n".join(lines[1:]))
corrections = payload.get("corrections")

if not isinstance(corrections, list):
    raise SystemExit("corrections must be a list")

if not corrections:
    raise SystemExit("Expected at least one correction")

for correction in corrections:
    for field in ("op", "start", "end", "before", "after"):
        if field not in correction:
            raise SystemExit(f"Missing correction field: {field}")

print("Smoke test passed")
print("Corrections:", len(corrections))
print("First correction:", json.dumps(corrections[0], ensure_ascii=True))
PY

docker exec "${container_id}" sh -lc "python - <<'PY'
import json

with open('/tmp/test.segments.json', 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

base_segments = payload['segments']
long_segments = []

for repeat in range(40):
    time_offset = repeat * 10.0
    for segment in base_segments:
        words = []
        for word in segment['words']:
            words.append(
                {
                    'start': round(word['start'] + time_offset, 3),
                    'end': round(word['end'] + time_offset, 3),
                    'word': word['word'],
                }
            )
        long_segments.append(
            {
                'id': len(long_segments),
                'start': round(segment['start'] + time_offset, 3),
                'end': round(segment['end'] + time_offset, 3),
                'text': segment.get('text', ''),
                'words': words,
            }
        )

with open('/tmp/test.segments.long.json', 'w', encoding='utf-8') as handle:
    json.dump({'segments': long_segments, 'glossary': payload.get('glossary', [])}, handle)
PY"

docker exec "${container_id}" sh -lc "python - <<'PY' > /tmp/smoke-response-long.json
import json
import httpx

with open('/tmp/test.segments.long.json', 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

response = httpx.post(
    'http://127.0.0.1:8080/correct',
    json=payload,
    timeout=300,
)

print(response.status_code)
print(response.text)
PY"

docker cp "${container_id}:/tmp/smoke-response-long.json" "${response_file}"

python3 - <<'PY' "${response_file}"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    lines = handle.read().splitlines()

if not lines:
    raise SystemExit("Long smoke response was empty")

status_code = lines[0].strip()
if status_code != "200":
    raise SystemExit(f"Long smoke request failed with HTTP {status_code}")

payload = json.loads("\n".join(lines[1:]))
corrections = payload.get("corrections")

if not isinstance(corrections, list):
    raise SystemExit("Long corrections must be a list")

if not corrections:
    raise SystemExit("Expected at least one correction for long payload")

print("Long smoke test passed")
print("Long corrections:", len(corrections))
PY

docker exec "${container_id}" sh -lc "python - <<'PY'
import json

with open('/tmp/test.segments.json', 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

payload['glossary'].update(
    {
        f'Placeholder Term {index}': f'Placeholder cue {index}'
        for index in range(180)
    }
)

with open('/tmp/test.segments.glossary-heavy.json', 'w', encoding='utf-8') as handle:
    json.dump(payload, handle)
PY"

docker exec "${container_id}" sh -lc "python - <<'PY' > /tmp/smoke-response-glossary-heavy.json
import json
import httpx

with open('/tmp/test.segments.glossary-heavy.json', 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

response = httpx.post(
    'http://127.0.0.1:8080/correct',
    json=payload,
    timeout=300,
)

print(response.status_code)
print(response.text)
PY"

docker cp "${container_id}:/tmp/smoke-response-glossary-heavy.json" "${response_file}"

python3 - <<'PY' "${response_file}"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    lines = handle.read().splitlines()

if not lines:
    raise SystemExit("Glossary-heavy smoke response was empty")

status_code = lines[0].strip()
if status_code != "200":
    raise SystemExit(f"Glossary-heavy smoke request failed with HTTP {status_code}")

payload = json.loads("\n".join(lines[1:]))
corrections = payload.get("corrections")

if not isinstance(corrections, list):
    raise SystemExit("Glossary-heavy corrections must be a list")

if not corrections:
    raise SystemExit("Expected at least one correction for glossary-heavy payload")

print("Glossary-heavy smoke test passed")
print("Glossary-heavy corrections:", len(corrections))
PY

rm -f "${response_file}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspaces/asr

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

for cmd in docker rg fd jq yq uv python3 node npm go cargo rustc shellcheck shfmt tmux just git gh nvcc nvidia-smi; do
  command -v "${cmd}" >/dev/null
done

python3 --version
node --version
go version
rustc --version
gh --version | head -n 1
nvcc --version | tail -n 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker compose version
uv --version
test -S /var/run/docker.sock
docker ps >/dev/null
docker run --rm alpine:3.22 true
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ -f requirements-dev.txt ]]; then
  uv pip install --system -r requirements-dev.txt
fi

pytest -q -o cache_dir=/tmp/.pytest_cache "${PROJECT_ROOT}/tests"

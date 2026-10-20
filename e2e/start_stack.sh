#!/usr/bin/env bash
# Starts the bot and the services it needs from the checked-out source, for
# e2e/run_checks.py. Each runs in the background, logging to e2e/logs/, and
# e2e/stop_stack.sh stops them. Linux only (the CI runner).
#
# Started here: Redis (empty, local), the speech service, the voice sidecar,
# core itself and, with E2E_LOCAL_DIFFUSION=1, the PR's image service on the
# CPU with a tiny model. The LLM and (by default) the image service are the
# real ones at E2E_LLM_HOST / E2E_DIFFUSION_URL: they are what a dependency
# bump has to keep working with.
#
# Required: E2E_DISCORD_TOKEN, E2E_LLM_HOST, E2E_MODEL, and E2E_DIFFUSION_URL
# unless E2E_LOCAL_DIFFUSION=1.
# Optional: CORE_PYTHON / SPEECH_PYTHON / DIFFUSION_PYTHON (interpreters for
# each service's venv; default python3), E2E_CACHE (model + package caches),
# E2E_TINY_IMAGE_MODEL.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOGS="$ROOT/e2e/logs"
WORK="$ROOT/.e2e"
CACHE="${E2E_CACHE:-$HOME/.cache/e2e}"
mkdir -p "$LOGS" "$WORK" "$CACHE/hf"
: > "$LOGS/pids"

for var in E2E_DISCORD_TOKEN E2E_LLM_HOST E2E_MODEL; do
  if [ -z "${!var:-}" ]; then echo "$var is not set" >&2; exit 1; fi
done
if [ "${E2E_LOCAL_DIFFUSION:-0}" != 1 ] && [ -z "${E2E_DIFFUSION_URL:-}" ]; then
  echo "E2E_DIFFUSION_URL is not set (or set E2E_LOCAL_DIFFUSION=1)" >&2; exit 1
fi

# start NAME DIR COMMAND...: run in the background, log to logs/NAME.log.
start() {
  local name=$1 dir=$2; shift 2
  (cd "$dir" && exec setsid "$@" > "$LOGS/$name.log" 2>&1) &
  echo "$name $!" >> "$LOGS/pids"
}

# wait_http NAME URL SECONDS: until URL answers 200 (models download on the
# first run, so the timeouts are long).
wait_http() {
  local name=$1 url=$2 deadline=$((SECONDS + $3))
  until curl -fsS -o /dev/null "$url"; do
    if [ $SECONDS -ge $deadline ]; then
      echo "$name did not become healthy at $url" >&2
      tail -n 50 "$LOGS/$name.log" >&2 || true
      exit 1
    fi
    sleep 3
  done
  echo "$name is up"
}

venv() {  # venv NAME PYTHON REQUIREMENTS...
  local dir="$WORK/$1" python=$2; shift 2
  "$python" -m venv "$dir"
  "$dir/bin/pip" install --quiet --upgrade pip
  "$dir/bin/pip" install --quiet "$@"
}

# -- Redis: empty for every run, on loopback only
if ! command -v redis-server > /dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq redis-server > /dev/null
fi
redis-server --port 6379 --bind 127.0.0.1 --save "" --appendonly no \
  --daemonize yes --logfile "$LOGS/redis.log"

# -- Speech service (speech-to-text and text-to-speech, CPU)
venv speech "${SPEECH_PYTHON:-python3}" -r "$ROOT/speechservice/requirements.txt"
HF_HOME="$CACHE/hf" start speech "$ROOT/speechservice" \
  "$WORK/speech/bin/python" -m uvicorn main:app --host 127.0.0.1 --port 8002

# -- Voice sidecar (Discord voice I/O)
BRIDGE_TOKEN=$(head -c 24 /dev/urandom | base64 | tr -d '/+=')
(cd "$ROOT/voicesidecar" && npm ci --omit=dev --ignore-scripts --no-audit --no-fund --silent)
VOICE_BRIDGE_HOST=127.0.0.1 VOICE_BRIDGE_PORT=8765 VOICE_BRIDGE_TOKEN="$BRIDGE_TOKEN" \
  SPEECH_URL=http://127.0.0.1:8002 VOICE_DEBUG=1 \
  start voice "$ROOT/voicesidecar" node src/index.js

# -- Image service: the PR's own copy on the CPU, only when it changed
DIFFUSION_URL="${E2E_DIFFUSION_URL:-}"
if [ "${E2E_LOCAL_DIFFUSION:-0}" = 1 ]; then
  # Same torch release as the image (its Dockerfile), CPU build.
  torch=$(grep -oE 'torch==[0-9][0-9.]*' "$ROOT/diffusionservice/Dockerfile" | head -n 1)
  venv diffusion "${DIFFUSION_PYTHON:-python3}" "$torch" --index-url https://download.pytorch.org/whl/cpu
  "$WORK/diffusion/bin/pip" install --quiet -r "$ROOT/diffusionservice/requirements.txt"
  HF_HOME="$CACHE/hf" IMAGE_MODEL="${E2E_TINY_IMAGE_MODEL:-hf-internal-testing/tiny-stable-diffusion-xl-pipe}" \
    IMAGE_STEPS=2 IMAGE_WIDTH=128 IMAGE_HEIGHT=128 \
    start diffusion "$ROOT/diffusionservice" \
    "$WORK/diffusion/bin/python" -m uvicorn main:app --host 127.0.0.1 --port 8001
  DIFFUSION_URL=http://127.0.0.1:8001
fi

# -- Core: the bot itself, with the PR's code and dependencies
venv core "${CORE_PYTHON:-python3}" -r "$ROOT/core/requirements.txt"
wait_http speech http://127.0.0.1:8002/health 900
if [ "${E2E_LOCAL_DIFFUSION:-0}" = 1 ]; then
  wait_http diffusion http://127.0.0.1:8001/health 900
fi
(
  # A subshell, so only core gets these settings, under core's own names.
  export DISCORD_TOKEN="$E2E_DISCORD_TOKEN"
  unset E2E_DISCORD_TOKEN E2E_WEBHOOK_URL
  export REDIS_HOST=127.0.0.1
  export LLM_HOST="$E2E_LLM_HOST" MODEL="$E2E_MODEL" DIFFUSION_URL="$DIFFUSION_URL"
  # A tiny model's picture never passes the vision check: check, don't retry.
  if [ "${E2E_LOCAL_DIFFUSION:-0}" = 1 ]; then export IMAGE_REVIEW_RETRIES=0; fi
  export SANDBOX_ENABLED=0 CONTENT_GUARD_ENABLED=0 WHATS_NEW_ENABLED=0
  export DOUBLE_REPLY_CHANCE=0 REASONING_EFFORT=low METRICS_PORT=9464 LOG_LEVEL=INFO
  export VOICE_ENABLED=1 VOICE_DEV_INJECT=1 VOICE_DEBUG=1 VOICE_IDLE_LEAVE_SECONDS=600
  export VOICE_BRIDGE_URL=ws://127.0.0.1:8765 VOICE_BRIDGE_TOKEN="$BRIDGE_TOKEN"
  export SPEECH_URL=http://127.0.0.1:8002
  start core "$ROOT/core" "$WORK/core/bin/python" -W ignore -u main.py
)
echo "Stack started; logs in $LOGS"

#!/usr/bin/env bash
# Stops what e2e/start_stack.sh started. Safe to run more than once.
set -uo pipefail

LOGS="$(cd "$(dirname "$0")" && pwd)/logs"
if [ -f "$LOGS/pids" ]; then
  while read -r name pid; do
    # setsid made each service its own process group: stop the whole group.
    kill -TERM -- "-$pid" 2> /dev/null || kill -TERM "$pid" 2> /dev/null || true
  done < "$LOGS/pids"
  : > "$LOGS/pids"
fi
redis-cli -h 127.0.0.1 shutdown nosave 2> /dev/null || true

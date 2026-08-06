#!/usr/bin/env bash
# Report how the router EPP split N requests across the leaf "clusters".
#
#   CONCURRENCY=16 ./route_split.sh 200 mc-a mc-b
#
# Counts are read off each leaf EPP's own request histogram, so this measures
# where traffic actually landed rather than what the router logged.
#
# CONCURRENCY matters: the queue/KV scorers only discriminate once requests
# queue up. Sequential load leaves every cluster idle and scoring 0, so the
# picker ties and the split looks random no matter which arm is deployed.
set -euo pipefail

N="${1:-100}"; shift || true
CONCURRENCY="${CONCURRENCY:-1}"
ROUTER_NS="${ROUTER_NS:-mc-router}"
RELEASE="${RELEASE:-mc-router}"
MODEL="${MODEL:-facebook/opt-125m}"
MAX_TOKENS="${MAX_TOKENS:-4}"
[[ $# -ge 1 ]] || { echo "usage: $0 <n> <leaf-ns> [leaf-ns...]" >&2; exit 1; }

pf_pids=(); pf_logs=()
cleanup() {
  for p in "${pf_pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
  for f in "${pf_logs[@]:-}"; do rm -f "$f"; done
}
trap cleanup EXIT

# Let kubectl pick the local port and read it back, so concurrent runs and
# already-bound ports never collide. Also replaces a blind sleep.
pf() { # ns svc remote-port -> local port on stdout
  local log; log=$(mktemp); pf_logs+=("$log")
  kubectl -n "$1" port-forward "svc/$2" ":$3" >"$log" 2>&1 &
  pf_pids+=($!)
  for _ in $(seq 100); do
    local lp; lp=$(sed -n 's/.*127\.0\.0\.1:\([0-9][0-9]*\).*/\1/p' "$log" | head -1)
    [[ -n $lp ]] && { echo "$lp"; return 0; }
    sleep 0.2
  done
  echo "port-forward to $2 in $1 failed:" >&2; cat "$log" >&2; return 1
}

router_port=$(pf "$ROUTER_NS" "${RELEASE}-epp" 8081)

declare -A port
for ns in "$@"; do
  epp=$(kubectl -n "$ns" get svc -o name | grep -- '-router-epp$' | head -1 | cut -d/ -f2)
  [[ -n $epp ]] || { echo "$ns: no EPP service found" >&2; exit 1; }
  port[$ns]=$(pf "$ns" "$epp" 9090)
done

# _count is the histogram's request tally; sum over label sets.
count() { curl -s "http://127.0.0.1:${port[$1]}/metrics" \
  | awk '/^llm_d_epp_request_duration_seconds_count/ {s+=$NF} END {print s+0}'; }

declare -A before
for ns in "$@"; do before[$ns]=$(count "$ns"); done

seq 1 "$N" | xargs -P "$CONCURRENCY" -I{} \
  curl -s -o /dev/null -X POST "http://127.0.0.1:${router_port}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"req {}\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0}"

w=0; for ns in "$@"; do (( ${#ns} > w )) && w=${#ns}; done
echo "routed $N requests:"
for ns in "$@"; do
  d=$(( $(count "$ns") - ${before[$ns]} ))
  printf "  %-${w}s %5d  (%d%%)\n" "$ns" "$d" $(( N > 0 ? d * 100 / N : 0 ))
done

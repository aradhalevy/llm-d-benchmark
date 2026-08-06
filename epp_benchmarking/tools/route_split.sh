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
MODEL="${MODEL:-facebook/opt-125m}"
MAX_TOKENS="${MAX_TOKENS:-4}"
[[ $# -ge 1 ]] || { echo "usage: $0 <n> <leaf-ns> [leaf-ns...]" >&2; exit 1; }

pf_pids=()
cleanup() { for p in "${pf_pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

kubectl -n "$ROUTER_NS" port-forward svc/mc-router-epp 18081:8081 >/dev/null 2>&1 &
pf_pids+=($!)

declare -A port
p=19100
for ns in "$@"; do
  epp=$(kubectl -n "$ns" get svc -o name | grep -- '-router-epp$' | head -1 | cut -d/ -f2)
  kubectl -n "$ns" port-forward "svc/$epp" "$p:9090" >/dev/null 2>&1 &
  pf_pids+=($!)
  port[$ns]=$p; ((p++))
done
sleep 5

# _count is the histogram's request tally; sum over label sets.
count() { curl -s "http://127.0.0.1:${port[$1]}/metrics" \
  | awk '/^llm_d_epp_request_duration_seconds_count/ {s+=$NF} END {print s+0}'; }

declare -A before
for ns in "$@"; do before[$ns]=$(count "$ns"); done

seq 1 "$N" | xargs -P "$CONCURRENCY" -I{} \
  curl -s -o /dev/null -X POST http://127.0.0.1:18081/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"req {}\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0}"

echo "routed $N requests:"
for ns in "$@"; do
  d=$(( $(count "$ns") - ${before[$ns]} ))
  printf '  %-10s %4d  (%d%%)\n' "$ns" "$d" $(( d * 100 / N ))
done

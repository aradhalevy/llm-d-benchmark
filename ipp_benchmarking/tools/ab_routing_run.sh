#!/bin/bash
# Routing A/B: median-TTFT scorer (smart) vs single-model runs. One arm/invocation.
# Configure the IPP for the arm per the README (helm upgrade -f <values>) first.
#   ab_routing_run.sh <arm> <ipp_values_file> [profile=half_8b.yaml]
set -u
ARM="${1:?arm}"; CFG="${2:?ipp values file}"; PROFILE="${3:-half_8b.yaml}"
REPO=/home/arad/new_git_repos/benchmark_try/llm-d-benchmark
NS=llm-d-arad; DAP=access-to-harness-data-workload-pvc
GW=http://infra-llmdbench-inference-gateway-istio.$NS.svc.cluster.local:80
D=ipp_benchmarking/example_outputs/ocp-research-agent-routing
DEST=$REPO/$D/$ARM
OCD=$DEST/oc-logs; STOP=/tmp/abr_${ARM}_stop
cd "$REPO"; source .venv/bin/activate 2>/dev/null; source .env 2>/dev/null
mkdir -p "$DEST/live-snapshots" "$OCD"
rm -f "$STOP"; : > "$DEST/decisions_rolling.log"; : > "$DEST/decisions_follow.log"

# 1. IPP must already be configured for this arm (helm upgrade -f $CFG), per the README.
# No timeouts: lift the per-route 30s request timeout (the sole load-shedder) so requests
# complete instead of recording 504s. Survives across runs; re-applied per arm to be safe.
for r in $(oc get httproute -n $NS -o name | grep -E 'qwen3-(8b|32b)'); do
  oc patch "$r" -n $NS --type=json -p='[{"op":"replace","path":"/spec/rules/0/timeouts/request","value":"1200s"}]' 2>/dev/null
done
POD=$(oc get pods -n $NS --no-headers | grep payload-processor | grep Running | awk '{print $1}' | head -1)
D8=$(oc get pods -n $NS --no-headers | grep 'qwen3-8b-decode' | grep Running | awk '{print $1}' | head -1)
D32=$(oc get pods -n $NS --no-headers | grep 'wen3-32b-decode' | grep Running | awk '{print $1}' | head -1)
echo "ARM=$ARM CFG=$CFG IPP=$POD"
oc delete pod -n $NS $DAP --force --grace-period=0 2>/dev/null   # let llmdbenchmark own the data-access pod

# 2. capture nets
( while [ ! -f "$STOP" ]; do oc logs -n $NS "$POD" --since=20s 2>/dev/null | grep -E '"msg":"Model selected"' >> "$DEST/decisions_rolling.log"; sleep 12; done
  oc logs -n $NS "$POD" --since=40s 2>/dev/null | grep -E '"msg":"Model selected"' >> "$DEST/decisions_rolling.log" ) & N1=$!
( while [ ! -f "$STOP" ]; do oc logs -f -n $NS "$POD" --since=5s 2>/dev/null | grep --line-buffered -E '"msg":"Model selected"' >> "$DEST/decisions_follow.log"; done ) & N2=$!
echo "nets rolling=$N1 follow=$N2"

# 3. summarizer harness (no planner)
nohup llmdbenchmark --spec cicd/ocp-qwen3-8b-32b-summarizer run -l inference-perf -w "$PROFILE" -p $NS > /tmp/abr_${ARM}_summ.log 2>&1 & SUMM=$!
echo "summarizer=$SUMM"

# 4. warm BOTH backends through the cold-start window (once llmdbenchmark creates the DAP pod),
#    so the scorer isn't cold at stage 0; stop when the summarizer's own load takes over.
for i in $(seq 1 48); do
  [ "$(oc get pod -n $NS $DAP --no-headers 2>/dev/null | awk '{print $2,$3}')" = "1/1 Running" ] && oc exec -n $NS $DAP -- true 2>/dev/null && break
  kill -0 "$SUMM" 2>/dev/null || break; sleep 5
done
WSTOP=/tmp/abr_${ARM}_warmstop; rm -f "$WSTOP"
( while [ ! -f "$WSTOP" ]; do
    for m in "Qwen/Qwen3-8B" "Qwen/Qwen3-32B"; do
      ( oc exec -n $NS $DAP -- curl -s -o /dev/null -m 1200 -X POST "$GW/v1/completions" -H 'Content-Type: application/json' \
          -d "{\"model\":\"$m\",\"prompt\":\"warm the ema\",\"max_tokens\":8}" 2>/dev/null ) &
    done; wait; sleep 1
  done ) & WARM=$!
echo "both-backend warm trickle=$WARM"
for i in $(seq 1 72); do
  # ponytail: count picks of ANY model (not just 8B) so a single-model baseline run
  # also detects "load is up"; threshold 20/15s = real load started for either pool.
  n=$(oc logs -n $NS "$POD" --since=15s 2>/dev/null | grep -c '"msg":"Model selected"')
  [ "${n:-0}" -ge 20 ] && { echo "load up ($n picks/15s) -> stop warm"; break; }
  kill -0 "$SUMM" 2>/dev/null || break; sleep 5
done
touch "$WSTOP"; pkill -P $WARM 2>/dev/null; kill $WARM 2>/dev/null

# 5. wait for summarizer to finish
while ! grep -q "All pods completed successfully" /tmp/abr_${ARM}_summ.log 2>/dev/null; do
  kill -0 "$SUMM" 2>/dev/null || { echo "summarizer exited"; break; }
  cp "$DEST/decisions_rolling.log" "$DEST/live-snapshots/rolling.$(date +%H%M).log" 2>/dev/null
  sleep 30
done
echo "WAITER done"; sleep 15
touch "$STOP"; sleep 25; kill $N1 $N2 2>/dev/null
cat "$DEST/decisions_rolling.log" "$DEST/decisions_follow.log" > "$DEST/ipp-decisions.log"

# 6. stage files + slim
SWS=$(grep -oE '/tmp/workspace_llmdbench_[^/]+/arad-[0-9-]+' /tmp/abr_${ARM}_summ.log | head -1)
SDIR=$(find "$SWS" -name stage_0_lifecycle_metrics.json 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
cp "$SDIR"/stage_*_lifecycle_metrics.json "$SDIR"/summary_lifecycle_metrics.json "$DEST/" 2>/dev/null
SEXP=$(basename "$SDIR" 2>/dev/null | sed 's/_1$//')
oc cp ipp_benchmarking/tools/extract_per_request_slim.py $NS/$DAP:/tmp/extract.py >/dev/null 2>&1
oc exec -n $NS $DAP -- python3 /tmp/extract.py "/requests/${SEXP}_1/per_request_lifecycle_metrics.json" /tmp/slim.json 2>&1
oc cp $NS/$DAP:/tmp/slim.json "$DEST/per_request_slim.json" >/dev/null 2>&1

# 7. logs + routing analysis (no planner -> planner_n=0)
oc logs -n $NS "$POD" --tail=-1 > "$OCD/ipp-tail.log" 2>/dev/null
oc logs -n $NS "$D8" -c vllm --tail=-1 > "$OCD/decode-8b-vllm.log" 2>/dev/null
oc logs -n $NS "$D32" -c vllm --tail=-1 > "$OCD/decode-32b-vllm.log" 2>/dev/null
oc delete pods -n $NS -l app=llmdbench-harness-launcher --force --grace-period=0 2>/dev/null
NAMESPACE=$NS bash ipp_benchmarking/collect_logs.sh > /tmp/abr_${ARM}_collect.log 2>&1
BUN=$(grep -oE 'collected-logs-[0-9]+' /tmp/abr_${ARM}_collect.log | head -1)
[ -n "$BUN" ] && [ -d "$REPO/$BUN" ] && mv "$REPO/$BUN" "$REPO/collected-logs-routing-$ARM" 2>/dev/null

{ echo "=== ARM $ARM (summarizer only, no planner; $CFG) ==="
  python3 ipp_benchmarking/tools/analyze_routing.py "$DEST/ipp-decisions.log" 0 0 0; } > "$DEST/routing-summary.txt" 2>&1
cat "$DEST/routing-summary.txt"
echo "ABR_DONE_$ARM"

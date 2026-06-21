#!/bin/bash
# A/B blog run — ONE arm per invocation. Planner (pinned 32B) + summarizer (profile decides
# smart-array vs static-8B). Path-1 capture: TWO independent decision nets (rolling --since
# snapshots + long-lived `oc logs -f`, unioned & deduped by x-request-id) so the IPP
# response-chunk flood can't drop decisions. Plus heavy OpenShift log capture.
#
#   ab_blog_run.sh <arm_name> <summarizer_profile.yaml>
# e.g. ab_blog_run.sh arm_b_static summarization_static_8b.yaml
set -u
ARM="${1:?arm name}"; PROFILE="${2:?summarizer profile}"
REPO=/home/arad/new_git_repos/new_benchmark/llm-d-benchmark
NS=llm-d-arad
DAP=access-to-harness-data-workload-pvc
GW=http://infra-llmdbench-inference-gateway-istio.llm-d-arad.svc.cluster.local:80
DEST=$REPO/ipp_benchmarking/example_outputs/ocp-research-agent-blog/$ARM
OCD=$DEST/oc-logs
STOP=/tmp/ab_${ARM}_stop
cd "$REPO"; source .venv/bin/activate 2>/dev/null; source .env 2>/dev/null
mkdir -p "$DEST/live-snapshots" "$OCD"
rm -f "$STOP" /tmp/ab_${ARM}_planner.txt
: > "$DEST/decisions_rolling.log"; : > "$DEST/decisions_follow.log"

POD=$(oc get pods -n $NS --no-headers | grep payload-processor | grep Running | awk '{print $1}' | head -1)
D8=$(oc get pods -n $NS --no-headers | grep 'qwen3-8b-decode' | grep Running | awk '{print $1}' | head -1)
D32=$(oc get pods -n $NS --no-headers | grep 'wen3-32b-decode' | grep Running | awk '{print $1}' | head -1)
E8=$(oc get pods -n $NS --no-headers | grep '8b-gaie-epp' | grep Running | awk '{print $1}' | head -1)
E32=$(oc get pods -n $NS --no-headers | grep '32b-gaie-epp' | grep Running | awk '{print $1}' | head -1)
GWP=$(oc get pods -n $NS --no-headers | grep 'inference-gateway-istio' | grep Running | awk '{print $1}' | head -1)
echo "ARM=$ARM PROFILE=$PROFILE IPP=$POD D8=$D8 D32=$D32"

# planner body (copied into the data-access pod AFTER llmdbenchmark creates it)
awk 'BEGIN{p="Decompose the research question into sub-tasks and synthesize the findings. "; s=""; for(i=0;i<150;i++)s=s p; printf "{\"model\":\"Qwen/Qwen3-32B\",\"prompt\":\"%s\",\"max_tokens\":256,\"ignore_eos\":true}", s}' > /tmp/pbody_${ARM}.json
# llmdbenchmark step 02 OWNS the data-access pod (it applies an exact spec; a hand-made pod of
# the same name fails on immutable fields). Remove any stale one so step 02 creates it clean.
oc delete pod -n $NS $DAP --force --grace-period=0 2>/dev/null

# NET 1: rolling snapshots (last 20s every 12s) -> survives rotation
( while [ ! -f "$STOP" ]; do
    oc logs -n $NS "$POD" --since=20s 2>/dev/null | grep -E '"msg":"Model selected"|"msg":"Starting model selection"' >> "$DEST/decisions_rolling.log"
    sleep 12
  done
  oc logs -n $NS "$POD" --since=40s 2>/dev/null | grep -E '"msg":"Model selected"' >> "$DEST/decisions_rolling.log" ) &
N1=$!
# NET 2: long-lived follow, restart on drop, filtered to decision lines
( while [ ! -f "$STOP" ]; do
    oc logs -f -n $NS "$POD" --since=5s 2>/dev/null | grep --line-buffered -E '"msg":"Model selected"' >> "$DEST/decisions_follow.log"
  done ) &
N2=$!
echo "capture nets: rolling=$N1 follow=$N2"

# summarizer harness (its step 02 creates the data-access pod $DAP)
nohup llmdbenchmark --spec cicd/ocp-qwen3-8b-32b-summarizer run -l inference-perf \
  -w "$PROFILE" -p $NS > /tmp/ab_${ARM}_summ.log 2>&1 &
SUMM=$!
echo "summarizer=$SUMM profile=$PROFILE"

# wait for llmdbenchmark to create $DAP, then start the planner against it
for i in $(seq 1 48); do
  s=$(oc get pod -n $NS $DAP --no-headers 2>/dev/null | awk '{print $2,$3}')
  [ "$s" = "1/1 Running" ] && oc exec -n $NS $DAP -- true 2>/dev/null && break
  kill -0 "$SUMM" 2>/dev/null || { echo "summarizer died before data-access pod ready"; break; }
  sleep 5
done
oc cp /tmp/pbody_${ARM}.json $NS/$DAP:/tmp/pbody.json >/dev/null 2>&1
echo "data-access pod ready -> starting planner"
# planner curl loop (~3 concurrent single-32B) until stop.
# log "<epoch_done> <http_code> <time_total>" so we can plot planner LATENCY over the run and
# bin it by the summarizer concurrency stage active at that wall-clock time (timeouts -> 504
# with time_total ~= the gateway ceiling, so they land in the timeout band of the latency plot).
( : > /tmp/ab_${ARM}_planner.txt
  while [ ! -f "$STOP" ]; do
    for c in 1 2 3; do
      ( r=$(oc exec -n $NS $DAP -- curl -s -o /dev/null -w '%{http_code} %{time_total}' -X POST "$GW/v1/completions" \
          -H 'Content-Type: application/json' -d @/tmp/pbody.json 2>/dev/null); echo "$(date +%s) $r" >> /tmp/ab_${ARM}_planner.txt ) &
    done
    wait
  done ) &
PLN=$!
echo "planner=$PLN"

while ! grep -q "All pods completed successfully" /tmp/ab_${ARM}_summ.log 2>/dev/null; do
  kill -0 "$SUMM" 2>/dev/null || { echo "summarizer exited"; break; }
  # periodic insurance snapshot
  cp "$DEST/decisions_rolling.log" "$DEST/live-snapshots/rolling.$(date +%H%M).log" 2>/dev/null
  sleep 30
done
echo "WAITER: summarizer complete"; sleep 15

touch "$STOP"; sleep 25            # let final rolling sweep + planner drain
pkill -P $PLN 2>/dev/null; kill $PLN $N1 $N2 2>/dev/null
ps=$(awk '$2==200' /tmp/ab_${ARM}_planner.txt 2>/dev/null | wc -l); pt=$(wc -l < /tmp/ab_${ARM}_planner.txt 2>/dev/null); pf=$((pt-ps))
cp /tmp/ab_${ARM}_planner.txt "$DEST/planner-http-codes.txt"

# union decisions
cat "$DEST/decisions_rolling.log" "$DEST/decisions_follow.log" > "$DEST/ipp-decisions.log"
uniqids=$(grep -oE '"x-request-id":"[^"]+"' "$DEST/ipp-decisions.log" | sort -u | wc -l)

# summarizer stage files from workspace (small, flushed at pod-complete)
SWS=$(grep -oE '/tmp/workspace_llmdbench_[^/]+/arad-[0-9-]+' /tmp/ab_${ARM}_summ.log | head -1)
SDIR=$(find "$SWS" -name stage_0_lifecycle_metrics.json 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
cp "$SDIR"/stage_*_lifecycle_metrics.json "$SDIR"/summary_lifecycle_metrics.json "$DEST/" 2>/dev/null
EXP=$(basename "$SDIR" 2>/dev/null | sed 's/_1$//')   # inference-perf-<ts>-<hash>
echo "EXP=$EXP"

# heavy OpenShift logs
oc logs -n $NS "$D8"  -c vllm --tail=-1 > "$OCD/decode-8b-vllm.log"  2>/dev/null
oc logs -n $NS "$D32" -c vllm --tail=-1 > "$OCD/decode-32b-vllm.log" 2>/dev/null
oc logs -n $NS "$E8"  --tail=-1 > "$OCD/epp-8b.log"  2>/dev/null
oc logs -n $NS "$E32" --tail=-1 > "$OCD/epp-32b.log" 2>/dev/null
oc logs -n $NS "$GWP" --tail=-1 > "$OCD/gateway-istio.log" 2>/dev/null
oc logs -n $NS "$POD" --tail=-1 > "$OCD/ipp-tail.log" 2>/dev/null
oc get events -n $NS --sort-by=.lastTimestamp > "$OCD/events.txt" 2>/dev/null
oc get pods,deploy,inferencepool,httproute,cm -n $NS -o wide > "$OCD/objects.txt" 2>/dev/null
oc describe pod "$POD" "$D8" "$D32" -n $NS > "$OCD/describe.txt" 2>/dev/null

# slim per_request: extract IN-POD on the harness PVC (the workspace copy is truncated
# mid-flush; the PVC keeps the complete file under /requests/<exp>_1/). The DAP pod has
# python3, so we read 851MB locally in-pod and copy out only the KB-sized slim.
oc cp ipp_benchmarking/tools/extract_per_request_slim.py $NS/$DAP:/tmp/extract.py >/dev/null 2>&1
oc exec -n $NS $DAP -- python3 /tmp/extract.py "/requests/${EXP}_1/per_request_lifecycle_metrics.json" /tmp/slim_${ARM}.json 2>&1
oc cp $NS/$DAP:/tmp/slim_${ARM}.json "$DEST/per_request_slim.json" >/dev/null 2>&1

# full collect_logs bundle
oc delete pods -n $NS -l app=llmdbench-harness-launcher --force --grace-period=0 2>/dev/null
NAMESPACE=$NS bash ipp_benchmarking/collect_logs.sh > /tmp/ab_${ARM}_collect.log 2>&1
BUN=$(grep -oE 'collected-logs-[0-9]+' /tmp/ab_${ARM}_collect.log | head -1)
[ -n "$BUN" ] && [ -d "$REPO/$BUN" ] && mv "$REPO/$BUN" "$REPO/collected-logs-blog-$ARM" 2>/dev/null

# routing breakdown
{ echo "=== ARM $ARM  profile=$PROFILE ==="
  echo "PLANNER http: sent=$pt ok=$ps fail=$pf"
  echo "decision capture: rolling+follow union, $uniqids unique x-request-id"; echo
  python3 ipp_benchmarking/tools/analyze_routing.py "$DEST/ipp-decisions.log" "$pt" 0 0
} > "$DEST/routing-summary.txt" 2>&1
cat "$DEST/routing-summary.txt"
echo "AB_BLOG_DONE_$ARM"

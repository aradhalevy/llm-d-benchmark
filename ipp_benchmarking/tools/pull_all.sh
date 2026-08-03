#!/bin/bash
# Pull an ENTIRE llmdbenchmark experiment dir off a workload-pvc, verified.
# The huge per_request_lifecycle_metrics.json goes separately (retry_gz); everything else -- stages,
# summary, stdout/stderr, run_metadata, config, profile, metrics/, analysis/, benchmark_report* --
# comes as one tarball.  oc truncates large transfers at random, so both are retried until valid.
#   pull_all.sh <ns> <experiment-dir-name> <dest>
set -u
NS=$1 EXP=$2 DEST=$3
R=$(cd "$(dirname "$0")/../.." && pwd)
DAP=access-to-harness-data-workload-pvc RP="/requests/${EXP}_1" BIG=per_request_lifecycle_metrics.json
mkdir -p "$DEST"

for i in $(seq 6); do
  oc exec -n "$NS" "$DAP" -- tar czf - -C "$RP" --exclude="$BIG" . > "$DEST/experiment_bundle.tar.gz" 2>/dev/null
  tar tzf "$DEST/experiment_bundle.tar.gz" >/dev/null 2>&1 && { echo "bundle OK try=$i $(stat -c%s "$DEST/experiment_bundle.tar.gz")"; break; }
  echo "bundle try=$i truncated at $(stat -c%s "$DEST/experiment_bundle.tar.gz")"
  [ "$i" = 6 ] && { echo "BUNDLE FAILED"; exit 1; }
done
tar xzf "$DEST/experiment_bundle.tar.gz" -C "$DEST" && rm -f "$DEST/experiment_bundle.tar.gz"

oc cp "$R/ipp_benchmarking/tools/extract_per_request_slim.py" "$NS/$DAP:/tmp/extract.py" >/dev/null 2>&1
oc exec -n "$NS" "$DAP" -- python3 /tmp/extract.py "$RP/$BIG" "/tmp/slim_${EXP}.json"
oc cp "$NS/$DAP:/tmp/slim_${EXP}.json" "$DEST/per_request_slim.json" 2>/dev/null

bash "$R/ipp_benchmarking/tools/retry_gz.sh" "$NS" "$RP/$BIG" "$DEST/$BIG.gz" 8
echo "PULL_ALL_DONE $NS/$EXP -> $DEST"

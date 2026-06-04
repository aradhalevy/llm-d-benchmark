#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llmdbench}"

id=1
while [[ -d "./collected-logs-${id}" ]]; do (( id++ )); done
LOG_DIR="./collected-logs-${id}"
mkdir -p "${LOG_DIR}"

for sel in \
  'app=payload-processor' \
  'llm-d.ai/inference-serving=true' \
  'inference.networking.k8s.io/igw-mode=inferencepool' \
  'gateway.networking.k8s.io/gateway-class-name'; do
  kubectl get pods -n "${NAMESPACE}" -l "${sel}" -o name 2>/dev/null | while read -r pod; do
    kubectl logs -n "${NAMESPACE}" "${pod}" --timestamps --previous 2>/dev/null \
      >"${LOG_DIR}/${pod#pod/}.log" || \
    kubectl logs -n "${NAMESPACE}" "${pod}" --timestamps \
      >"${LOG_DIR}/${pod#pod/}.log" 2>&1 || true
  done
done

# Copy benchmark run results from the local llmdbenchmark workspace.
#
# Resolution order (matches llmdbenchmark itself):
#   1. LLMDBENCH_WORKSPACE env var
#   2. ~/data/kind-sim-multi  (workDir from config/scenarios/cicd/kind-sim-multi.yaml)
#   3. Most-recently-modified /tmp/workspace_llmdbench_* directory
_find_workspace() {
  if [[ -n "${LLMDBENCH_WORKSPACE:-}" && -d "${LLMDBENCH_WORKSPACE}" ]]; then
    echo "${LLMDBENCH_WORKSPACE}"
    return
  fi
  local default_workdir="${HOME}/data/kind-sim-multi"
  if [[ -d "${default_workdir}" ]]; then
    echo "${default_workdir}"
    return
  fi
  # Fall back to the most recently modified tmp workspace
  local latest
  latest=$(find /tmp -maxdepth 1 -name "workspace_llmdbench_*" -type d \
    -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
  echo "${latest:-}"
}

WORKSPACE=$(_find_workspace)
if [[ -n "${WORKSPACE}" ]]; then
  # Each llmdbenchmark run creates a timestamped subdir inside the workspace.
  # Find the most recently modified one that has a results/ or analysis/ child.
  RUN_SUBDIR=""
  while IFS= read -r d; do
    if [[ -d "${d}/results" || -d "${d}/analysis" ]]; then
      RUN_SUBDIR="${d}"
      break
    fi
  done < <(find "${WORKSPACE}" -maxdepth 1 -mindepth 1 -type d \
             -printf "%T@ %p\n" 2>/dev/null | sort -rn | awk '{print $2}')

  if [[ -n "${RUN_SUBDIR}" ]]; then
    BENCH_RESULTS_DIR="${LOG_DIR}/benchmark-results"
    mkdir -p "${BENCH_RESULTS_DIR}"
    copied=0
    for subdir in results analysis; do
      src="${RUN_SUBDIR}/${subdir}"
      if [[ -d "${src}" ]]; then
        cp -r "${src}" "${BENCH_RESULTS_DIR}/${subdir}"
        copied=$((copied + 1))
      fi
    done
    echo "Benchmark results copied from ${RUN_SUBDIR} to ${BENCH_RESULTS_DIR}/"
  else
    echo "Workspace found at ${WORKSPACE} but no run subdirectory with results/ or analysis/ present; skipping benchmark results."
  fi
else
  echo "No llmdbenchmark workspace found (set LLMDBENCH_WORKSPACE or run from ~/data/kind-sim-multi); skipping benchmark results."
fi

# Best-effort: capture per-GPU DCGM metrics (tensor/SM/DRAM active, power) for
# the run window from OpenShift monitoring. No-op on clusters without the GPU
# Operator / Thanos (e.g. Kind). Never fails the collection.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/tools/collect_dcgm.py" ]]; then
  NAMESPACE="${NAMESPACE}" python3 "${_SCRIPT_DIR}/tools/collect_dcgm.py" \
    --logs-dir "${LOG_DIR}" --namespace "${NAMESPACE}" || \
    echo "collect_dcgm: skipped (non-fatal)."
fi

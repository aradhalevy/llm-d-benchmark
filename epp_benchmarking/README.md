# EPP multi-cluster benchmarking

Measure a router EPP whose candidate endpoints are whole clusters, scored on each
cluster's pool-aggregate KV-cache and queue metrics.

Peer clusters are simulated with one namespace per "cluster". Each leaf
namespace runs a normal llm-d stack; the router runs the standalone
(`epponly`) topology, so its Envoy forwards to whatever `IP:port` the EPP
picks.

```
load ──> router Envoy :8081 ──> router EPP (multicluster plugins)
                                  │ reads clusters.yaml (peer gateway ClusterIPs)
                                  │ scrapes each leaf EPP :9090/metrics
                                  └─> leaf gateway :80 ──> leaf EPP ──> vLLM
```

Troubleshooting and Gotchas are in [`CLAUDE.md`](./CLAUDE.md).

**Prerequisites:** `llmdbenchmark` installed and on your `PATH` — see
[Getting Started → Install](../README.md#install) in the repo README — plus
`helm` and `envsubst` (from GNU gettext; not installed by default on macOS),
and `kind` + `docker` for the Kind walkthrough. Every command below runs from
the repo root against your default kubeconfig.

## Kind (no GPU)

Two leaves with asymmetric capacity, same model, sim backends. `mc-a`, `mc-b`
and `mc-router` are just literals — rename them as long as you do so
consistently, remembering that the Helm release name determines the
`mc-router-epp` Deployment, Service and endpoint URL used later. The one name
that is *not* free is the `mc-clusters` ConfigMap: `router/values.yaml` mounts
it by name, so change it in both places or not at all.

### 1. Create the cluster and load images

Side-loading keeps standup from timing out on a cold image pull. The tags must
match what your `llmdbenchmark` version actually deploys — if standup still
stalls pulling, check `images.benchmark` in `config/templates/values/defaults.yaml`
and side-load that tag instead.

```bash
kind create cluster

for i in ghcr.io/llm-d/llm-d-benchmark:v0.7.0 \
         ghcr.io/llm-d/llm-d-inference-sim:v0.8.2 \
         ghcr.io/llm-d/llm-d-router-endpoint-picker:main; do
  docker pull "$i" && kind load docker-image "$i"
done
```

### 2. Stand up an llm-d stack in each namespace

After the images finished loading, we stand up one llm-d stack per namespace, asymmetric capacity so the router has something to decide.

```bash
llmdbenchmark --spec cicd/kind-sim-mc-leaf standup -p mc-a --set decode.replicas=1
llmdbenchmark --spec cicd/kind-sim-mc-leaf standup -p mc-b --set decode.replicas=3
```

### 3. Slow the sims down

Without this the sims answer instantly, nothing ever queues, and both arms score
identically — see "Load must saturate" in `CLAUDE.md`.

```bash
for ns in mc-a mc-b; do
  d=$(kubectl -n $ns get deploy -o name | grep decode)
  kubectl -n $ns patch $d --type=json -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args","value":["--model","facebook/opt-125m","--port","8200","--served-model-name","facebook/opt-125m","--time-to-first-token=2000","--inter-token-latency=100","--max-num-seqs=2"]}]'
done
```

### 4. Publish the cluster list

[`router/clusters.yaml`](./router/clusters.yaml) is what `multicluster-file-discovery`
reads. It is a template because ClusterIPs are only known once the leaves exist,
and they must be IPs rather than DNS names.

```bash
export MC_A_NS=mc-a MC_B_NS=mc-b

gw_ip()   { kubectl -n "$1" get svc -l gateway.networking.k8s.io/gateway-name -o jsonpath='{.items[0].spec.clusterIP}'; }
gw_port() { kubectl -n "$1" get svc -l gateway.networking.k8s.io/gateway-name -o jsonpath='{.items[0].spec.ports[?(@.port==80)].port}'; }
epp_ip()  { kubectl -n "$1" get "$(kubectl -n "$1" get svc -o name | grep -m1 -- '-router-epp$')" -o jsonpath='{.spec.clusterIP}'; }

export MC_A_GW_IP=$(gw_ip "$MC_A_NS") MC_A_GW_PORT=$(gw_port "$MC_A_NS") MC_A_EPP_IP=$(epp_ip "$MC_A_NS")
export MC_B_GW_IP=$(gw_ip "$MC_B_NS") MC_B_GW_PORT=$(gw_port "$MC_B_NS") MC_B_EPP_IP=$(epp_ip "$MC_B_NS")

kubectl create ns mc-router

clusters=$(envsubst < epp_benchmarking/router/clusters.yaml)
printf '%s\n' "$clusters"

# Any lookup that missed renders as an empty value, and file discovery skips
# that endpoint silently -- leaving a router that scores one cluster. Refuse to
# apply instead.
if printf '%s' "$clusters" | grep -qE '(address|metricsAddress): *$|port: ""'; then
  echo 'ERROR: empty value above -- are both leaves up in the namespaces you set?' >&2
else
  printf '%s' "$clusters" | kubectl -n mc-router create configmap mc-clusters \
    --from-file=clusters.yaml=/dev/stdin --dry-run=client -o yaml | kubectl apply -f -
fi
```

### 5. Install the router

The chart's default EPP image has no multicluster plugins;
[`router/values.yaml`](./router/values.yaml) pins a newer one that does, and
enables the Alpha plugin gate.

```bash
helm upgrade --install mc-router \
  oci://ghcr.io/llm-d/charts/llm-d-router-standalone --version v0.9.0 \
  -n mc-router -f epp_benchmarking/router/values.yaml
kubectl -n mc-router rollout status deploy/mc-router-epp --timeout=300s
```

### 6. Benchmark through the router

`--endpoint-url` points the harness at the router instead of a single stack, so
the load runs in-cluster and every request is routed by the multicluster
scorers.

```bash
epp_benchmarking/tools/analyze_split.py --save /tmp/before.json mc-a mc-b

llmdbenchmark --spec cicd/kind-sim-mc-leaf run -p mc-a -l inference-perf \
  -w mc_kind_concurrent.yaml \
  --endpoint-url http://mc-router-epp.mc-router.svc.cluster.local:8081
```

To compare against the no-scoring baseline, swap the picker and repeat step 6.
`clusters.yaml` is unchanged.

```bash
helm upgrade --install mc-router \
  oci://ghcr.io/llm-d/charts/llm-d-router-standalone --version v0.9.0 \
  -n mc-router -f epp_benchmarking/router/values.yaml \
  --set router.epp.pluginsConfigFile=mc-random.yaml
kubectl -n mc-router rollout restart deploy/mc-router-epp
kubectl -n mc-router rollout status deploy/mc-router-epp --timeout=300s
```

## OpenShift (GPU)

The same three namespaces with real Qwen3-8B, one GPU per decode pod — four for
the 1+3 layout below. Steps 4–6 above are reused as-is; only standup differs.

Stand the leaves up **before** the router: the standalone chart renders an
InferencePool, and the `inference.networking.k8s.io` CRD arrives with the first
leaf.

### 1. Stand up a leaf in each namespace

No image side-loading and no `kind create` — the cluster pulls normally and
`llmdbenchmark` creates the namespaces. Export `HF_TOKEN` if your cluster needs
one for Hugging Face; standup turns it into a Secret.

```bash
llmdbenchmark --spec cicd/ocp-mc-leaf standup -p <prefix>-mc-a --set decode.replicas=1
llmdbenchmark --spec cicd/ocp-mc-leaf standup -p <prefix>-mc-b --set decode.replicas=3
```

On a mixed cluster, pin the GPU model with
`--set decode.acceleratorType.labelKey=nvidia.com/gpu.product --set
decode.acceleratorType.labelValue=<product>`. Prefer this over
`affinity.nodeSelector`, which is ignored unless `affinity.enabled` is also set.

There is no equivalent of step 3: the servers run unthrottled and the pressure
comes from the request rate instead. See "Sizing the workload" in `CLAUDE.md`
for why the 1+3 asymmetry is what makes that enough.

### 2. Publish the cluster list

Run step 4's block with its first line replaced by the one below — the rest,
including the empty-value guard, is unchanged. The helpers read ClusterIPs, so
they work whether the leaf gateway Service is NodePort or LoadBalancer.

```bash
export MC_A_NS=<prefix>-mc-a MC_B_NS=<prefix>-mc-b MC_ROUTER_NS=<prefix>-mc-router
```

Substitute `$MC_ROUTER_NS` for the literal `mc-router` in step 4's
`create configmap` line too — on a shared cluster you rarely own the bare name.

### 3. Install the router

Step 5 plus a second values file that restores the chart's production sizing —
the Kind values throttle the Envoy to one core.

```bash
helm upgrade --install mc-router \
  oci://ghcr.io/llm-d/charts/llm-d-router-standalone --version v0.9.0 \
  -n "$MC_ROUTER_NS" \
  -f epp_benchmarking/router/values.yaml \
  -f epp_benchmarking/router/values-ocp.yaml
kubectl -n "$MC_ROUTER_NS" rollout status deploy/mc-router-epp --timeout=600s
```

### 4. Benchmark through the router

```bash
epp_benchmarking/tools/analyze_split.py --save /tmp/before.json "$MC_A_NS" "$MC_B_NS"

llmdbenchmark --spec cicd/ocp-mc-leaf run -p "$MC_A_NS" -l inference-perf \
  -w mc_saturation_poisson.yaml \
  --endpoint-url "http://mc-router-epp.$MC_ROUTER_NS.svc.cluster.local:8081"
```

The ladder is 45 minutes per arm, so budget an hour and a half for the pair.
Swap to the baseline arm as in Kind, keeping both `-f` files.

## Analysing a run

`tools/analyze_split.py` reports where traffic actually landed — counted from
each leaf EPP's own request histogram, not from what the router logged — plus
the latency and throughput the harness recorded. Pass the results directory
`llmdbenchmark run` printed as `Local results:`.

```bash
epp_benchmarking/tools/analyze_split.py \
  --since /tmp/before.json \
  --results <results-dir> \
  mc-a mc-b
```

It prints the per-namespace request counts and shares, then the run's request
count, wall clock, throughput, and mean/p90 TTFT and latency:

```
routed <n> requests
  mc-a     <n>   <pct>
  mc-b     <n>   <pct>

run summary (<treatment>)
  requests        <n> ok / <n> failed
  wall clock      <s>
  throughput      <req/s>, <out-tok/s>
  TTFT            mean <s>  p90 <s>
  latency         mean <s>  p90 <s>
```

Run it once per arm and compare: the scored arm should track the leaves' capacity
ratio while the `random-picker` baseline splits evenly.

The counters are cumulative, hence the `--save`/`--since` pair around each run.
Metrics are read through the API server's service proxy, so no port-forward is
needed.

### Latency over the ladder

`plot_e2e_timeseries.py` draws every request as a point at its arrival time with
a continuous p50 line across all stages, and labels each stage band with its
requested rate — so a latency knee can be read off the rung that caused it.
Needs `matplotlib`.

Extraction is a separate step because `per_request_lifecycle_metrics.json` is
multi-GB (2.9GB for 11k requests) and usually truncated mid-write; the extractor
recovers every complete record rather than failing on the tail.

```bash
run=<results-dir>/<run-subdir>
epp_benchmarking/tools/extract_per_request_slim.py \
  $run/per_request_lifecycle_metrics.json $run/per_request_slim.json
epp_benchmarking/tools/plot_e2e_timeseries.py $run --bin 15
```

`--bin` sets the median window in seconds; `--log` switches to a log y axis,
which is worth it when an arm collapses and the spread crosses two decades.

To put both arms on one axes, one colour per run:

```bash
epp_benchmarking/tools/plot_e2e_compare.py $smart_run $random_run \
  --labels scored,baseline --log
```

Its x axis is stage index rather than elapsed time: an arm that falls behind
stretches its stages, so real time would slide identical rungs out of alignment.
Each run is warped through its own stage windows; medians are still taken over
real `--bin` second windows.

Colouring points by *leaf* is not possible from harness data — the per-request
records carry no upstream identity, and both leaves answer to the same model
name. Per-leaf behaviour comes from the EPP gauges instead (`analyze_split.py`,
or `llm_d_epp_average_*` scraped over the run).

## Files

| path | what |
|---|---|
| `CLAUDE.md` | gotchas, workload sizing, design reasoning |
| `router/clusters.yaml` | peer cluster list; `envsubst` template (step 4) |
| `router/values.yaml` | router EPP chart values; both arms in `pluginsCustomConfig` |
| `router/values-ocp.yaml` | sizing overlay for the GPU run |
| `tools/analyze_split.py` | routing split + run summary |
| `tools/extract_per_request_slim.py` | slim per-request records from the multi-GB harness dump |
| `tools/plot_e2e_timeseries.py` | per-request latency scatter + p50 over the RPS ladder |
| `tools/plot_e2e_compare.py` | the same, two or more runs overlaid, one colour per run |
| `config/scenarios/cicd/kind-sim-mc-leaf.yaml` | leaf stack (sim), stood up once per namespace |
| `config/scenarios/cicd/ocp-mc-leaf.yaml` | leaf stack (Qwen3-8B, 1 GPU/pod) |
| `workload/profiles/inference-perf/mc_kind_concurrent.yaml.in` | saturating ladder for the sims |
| `workload/profiles/inference-perf/mc_saturation_poisson.yaml.in` | Poisson RPS ladder for the GPU leaves |

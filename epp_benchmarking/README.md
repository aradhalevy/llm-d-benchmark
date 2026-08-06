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

## Kind (no GPU)

Two leaves with asymmetric capacity, same model, sim backends.

```bash
kind create cluster --name mc
# A dedicated kubeconfig: `kind create cluster` switches the current context of
# your shared one, which is rude if something else is using it.
export KUBECONFIG=$(mktemp) && kind get kubeconfig --name mc > "$KUBECONFIG"

# Side-load: the benchmark image is large and the default 6m wait times out on a cold pull.
for i in ghcr.io/llm-d/llm-d-benchmark:v0.7.0 \
         ghcr.io/llm-d/llm-d-inference-sim:v0.8.2 \
         ghcr.io/llm-d/llm-d-router-endpoint-picker:main; do
  docker pull "$i" && kind load docker-image --name mc "$i"
done

./install.sh && source .venv/bin/activate   # needs sudo: it installs helm/helmfile/yq
llmdbenchmark --spec cicd/kind-sim-mc-leaf standup -p mc-a --set decode.replicas=1
llmdbenchmark --spec cicd/kind-sim-mc-leaf standup -p mc-b --set decode.replicas=3

./epp_benchmarking/tools/deploy_router.sh mc-a mc-b          # smart arm
CONCURRENCY=16 ./epp_benchmarking/tools/route_split.sh 200 mc-a mc-b

ARM=mc-random.yaml ./epp_benchmarking/tools/deploy_router.sh mc-a mc-b   # baseline
CONCURRENCY=16 ./epp_benchmarking/tools/route_split.sh 200 mc-a mc-b
```

Give the sims latency and a low concurrency cap first, or nothing queues and
both arms look identical (see "Load must be concurrent" below):

```bash
for ns in mc-a mc-b; do
  d=$(kubectl -n $ns get deploy -o name | grep decode)
  kubectl -n $ns patch $d --type=json -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args","value":["--model","facebook/opt-125m","--port","8200","--served-model-name","facebook/opt-125m","--time-to-first-token=2000","--inter-token-latency=100","--max-num-seqs=2"]}]'
done
```

### Measured (1 vs 3 decode pods, 200 requests, concurrency 16)

| arm | mc-a (1 pod) | mc-b (3 pods) | wall clock |
|---|---|---|---|
| smart (kv-cache + queue scorers) | 61 (30%) | 139 (69%) | 74s |
| baseline (`random-picker`) | 99 (49%) | 101 (50%) | 113s |

Scored routing tracks the 1:3 capacity ratio and finishes the same work 34%
faster; the baseline keeps overloading the single-pod cluster. A repeat of the
smart arm gave 57/143 (28%/71%), so run-to-run spread is a few points.

Namespaces, the Kind cluster name, and the router release name are all just
defaults — override with `ROUTER_NS`, `RELEASE`, `CHART_VERSION`, `MODEL`,
`MAX_TOKENS`, `CONCURRENCY`, `ARM`. Nothing is pinned to a machine or a user.

## Gotchas

These are all load-bearing — each one silently produces a no-op rather than an error.

- **Leaf metrics auth.** The EPP metrics endpoint defaults to
  `--metrics-endpoint-auth=true` and answers **401** to an anonymous scrape.
  `multicluster-metrics-data-source` supports TLS client certs but *no bearer
  token*, so the scrape fails and both scorers return no score. The leaf
  scenario sets `router.monitoring.prometheus.auth.enabled: false`.
- **Scheme.** The plugin defaults to `https`; a leaf EPP serves metrics over
  plain HTTP unless `metrics-cert-dir` is set. Hence `scheme: http`.
- **`schedulingProfiles` is not optional.** An empty list passes validation and
  yields a profile with no scorers — the EPP starts happily and scores nothing.
  The PR's own config example omits it.
- **Addresses must be IPs.** `ORIGINAL_DST` with `use_http_header` parses the
  header as `IP:port`. File discovery permits hostnames, but a DNS name will not
  resolve there. `deploy_router.sh` reads live ClusterIPs for this reason.
- **Image.** `v0.9.0` predates the PR and has none of the multicluster plugins;
  the router pins `tag: main`. All six types are registered **Alpha**, so
  `--allow-experimental-plugins` is required or the EPP refuses to start.
- **Load must be concurrent.** Both scorers read queue depth and KV usage. Under
  sequential load every cluster is idle and scores 0, `max-score-picker` ties,
  and the split looks random on *both* arms.
- The router EPP also auto-instantiates the stock `metrics-data-source` /
  `core-metrics-extractor` as a second datalayer poller. It scrapes the peer
  gateway port and logs failures; harmless, but it is noise in the logs.

## Files

| path | what |
|---|---|
| `config/scenarios/cicd/kind-sim-mc-leaf.yaml` | leaf stack, stood up once per namespace |
| `router/values.yaml` | router EPP chart values; both arms in `pluginsCustomConfig` |
| `tools/deploy_router.sh` | renders `clusters.yaml` from live ClusterIPs, installs the router |
| `tools/route_split.sh` | drives load, reports where it landed |

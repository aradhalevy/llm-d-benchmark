# EPP multi-cluster benchmarking -- notes

Findings, gotchas and dead ends for the cluster-scoped router EPP. Every item
here was hit and verified against a running cluster.

## Gotchas

Each of these silently produces a no-op rather than an error, which is what
makes them expensive.

- **Leaf metrics auth.** The EPP metrics endpoint defaults to
  `--metrics-endpoint-auth=true` and answers **401** to an anonymous scrape.
  `multicluster-metrics-data-source` supports TLS client certs but *no bearer
  token*, so the scrape fails, no attribute is written, and both scorers return
  no score. The leaf scenarios set `router.monitoring.prometheus.auth.enabled:
  false`, which the chart renders to `--metrics-endpoint-auth=false`.

- **Scheme.** `multicluster-metrics-data-source` defaults to `https` (it is
  meant to cross a trust boundary). A leaf EPP serves metrics over plain HTTP
  unless `metrics-cert-dir` is set, so the router config pins `scheme: http`.

- **`schedulingProfiles` is not optional.** An empty list passes validation and
  yields a profile with no scorers -- the EPP starts happily and scores nothing.
  The upstream config example omits it, so copying that example gives you a
  router that looks healthy and routes at random.

- **`address` must be an IP.** The router's Envoy resolves the picked endpoint
  through `ORIGINAL_DST` with `use_http_header`, which parses
  `x-gateway-destination-endpoint` as `IP:port`. File discovery itself accepts
  hostnames, so a Service DNS name loads fine and then every request fails with
  **503 "no healthy upstream"** (verified). `metricsAddress` is scraped by the
  EPP's own Go client, so a DNS name *would* work there.

- **ClusterIPs do not survive teardown/standup.** They are stable for a
  Service's lifetime only. Re-render `clusters.yaml` after rebuilding a leaf, or
  the router points at an address that no longer answers.

- **Image.** `v0.9.0` predates the multicluster plugins and contains none of
  them -- confirmed by extracting the binary and grepping for the plugin type
  strings. Use `tag: main`. All six variants are registered **Alpha**, so
  `--allow-experimental-plugins` is required or the EPP refuses to start.

- **Load must saturate.** Both scorers read queue depth and KV utilization.
  Below saturation every cluster reports 0, `max-score-picker` ties, and the
  split looks random on *both* arms -- so a too-gentle workload silently
  produces a null result rather than a failure.

- **Router before leaves (OpenShift).** The standalone chart renders an
  InferencePool of its own, so `inference.networking.k8s.io` must already exist.
  On Kind the leaf standup installs it; installing the router into a cluster
  with no llm-d stack yet fails on the missing CRD.

- **Helm deep-merges the two values files.** `-f values.yaml -f values-ocp.yaml`
  merges key by key, so a `limits` entry left out of the overlay silently keeps
  the Kind value. Omitting `cpu` there leaves the Envoy capped at 1 core while
  the memory limit reads 16Gi -- which looks deliberate. Restate every field.

- **`kubectl logs` loses most of a `--v=4` run.** kubelet rotates the container
  log at 10Mi and `kubectl logs` only reads the current file, so counts derived
  from it go *down* as a run proceeds; `logs -f` exits outright at each
  rotation. A 400-request arm read back as 250 routing decisions, and an
  in-cluster `logs -f` restart loop -- the obvious fix, and what the IPP
  experiment uses -- did no better: 28 of 400 TTFT observations, because the
  rotated bytes are gone before the follow reconnects. Only reading the node's
  rotated files recovers the run (`router/logtail.yaml`, 400/400). Filtering
  the noise is not an option either: the chatty debug callers are
  `handlers/response.go` and the metrics datasource, and the rotation has
  already happened by the time anything client-side sees a line.

- **`/debug/plugins/state` does not exist in file-discovery mode.** The handler
  is mounted on the controller-runtime manager, which the multicluster router
  never starts; it serves a standalone mux with `/metrics` and pprof only. Any
  plugin's `DumpState` is therefore unreachable here -- debug logs are the only
  window into scorer internals.

- **Always check captured requests against requests *offered*.** The logtail is
  not unconditionally lossless. It captured 400/400 on Kind and 45,360/45,360
  on the 2+3 GPU arms, but only 11,569/15,480 on the single-leaf calibration
  ladder -- and the loss was not spread out: stages 0-5 were 100% and the final
  24 req/s stage was **9.5%**. Container-log rotation during the highest-volume
  stage outruns the follower's reconnect, and the bytes are gone. Line counts
  look healthy while this happens, so compare `collect_epp_log.sh`'s request
  count against the ladder's offered total (sum of rate x duration), per stage
  if it matters. A shortfall concentrated in the last, hottest stage will
  silently flatter any per-stage conclusion drawn from it.

- **`ttft-aware-scorer` cold start dominates a short run.** Whichever endpoint
  crosses `minRequests` (default 10) first becomes `trusted`; the other still
  reads cold, scores 0, and gets only the `explorationRate` probes until it
  calibrates. Measured on a 400-request Kind arm: mc-b became trusted at
  request #107, and the split was 77/23 *the wrong way* before that and 18/82
  after. The run-level number is an average of the two, so two runs of the same
  config gave 24/76 and 40/60. Compare arms on the post-calibration segment, or
  make the run long enough that the warm-up is a small fraction of it.

- **Second datalayer poller.** The EPP auto-instantiates the stock
  `metrics-data-source` / `core-metrics-extractor` alongside the multicluster
  pair. It scrapes the peer gateway port and logs failures. Harmless, but it is
  noise when reading router logs.

## Never use closed-loop load

**Every profile is `load.type: poisson`. Do not write `type: concurrent` or
`concurrency_level`, anywhere, for any reason.**

A closed-loop client only sends a new request once an old one returns, so it
self-throttles and a backlog can never form. Queue depth and KV utilization --
the exact signals the multicluster scorers read -- then stay near zero however
slow a leaf gets, and every arm ties. Measured on H100s running Qwen3-8B: a
fixed-concurrency run at 256 with 512-token outputs peaked at **7% KV
utilization and zero queueing**. An 80GB card holds far more of an 8B model's
KV than a closed-loop harness ever offers it, because every request the server
has not answered is a request the harness has not sent.

Open-loop keeps offering requests at the stated rate, so a leaf that falls
behind accumulates backlog and becomes visible. This holds even when a
closed-loop sweep looks like the more direct probe of something load-dependent
(such as the TTFT curve, which is a function of in-flight count): it is not an
acceptable trade here.

## Sizing the workload

Getting this wrong is the next most likely reason a run produces a tie or fails
outright, and the right size is hardware-specific.

`mc_kind_poisson.yaml` is sized for the sims: deliberately slowed and capped at
`--max-num-seqs=2`, so a pod serves 2 at a time at ~5.2s each -- ~0.385 req/s
per pod, ~1.54 req/s for a 1+3 fleet, and the 1-pod leaf breaks first at ~0.77
req/s offered under an even split. Prompts must stay under the scenario's
`maxModelLen: 1024`, or every request 400s.

A GPU-sized ladder pointed at Kind sends thousands of requests at rates the sims
cannot approach, and the harness times out before finishing.

`mc_asym_poisson.yaml` targets the GPU leaves instead. The servers stay
unthrottled -- no `--max-num-seqs` cap, `gpuMemoryUtilization` at 0.90 -- so the
numbers remain a real capacity measurement.

The asymmetry is what makes the rates enough. Under an even split a 2+3 layout
puts half the fleet's rate on the 2-pod leaf, so it saturates before the fleet
does. The scored arm should close that gap; the `random-picker` arm cannot, and
its queue depth on the small leaf is the tell. Fleet-wide saturation is not
required and would mostly cost GPU hours.

The ladder climbs 12->60 req/s and back to 24, 180s per stage with `interval:
0`, so a stage's backlog carries into the next one. 45,360 requests, ~25 min per
arm. The descent is not decoration: a router that only sheds load one way looks
correct on the way up.

`total_count: 4000` means 4,000 distinct prompts are reused across those 45k
requests, so prefix-cache hits are common. That inflates throughput relative to
unique traffic, equally on both arms.

`per_request: false` is not optional at these rates: 45,360 records
OOM-killed the harness pod mid-write on the first attempt, leaving a 0-byte
`per_request_lifecycle_metrics.json`. The stage summaries survived, which is
what every plot here reads anyway.

## Load has to come from inside the cluster

Driving requests with `curl` through a `kubectl port-forward` cannot saturate
real GPUs -- the single proxied TCP connection, not the accelerator, becomes the
bottleneck. Use `llmdbenchmark run --endpoint-url <router>` so the harness pod
runs in-cluster.

## Why a namespace can stand in for a cluster

Every per-stack resource name derives from `sha256(namespace/model.name)`, so
two namespaces already yield distinct InferencePools, EPPs and Deployments for
the *same* `model.name`. None of the `-a`/`-b` model aliasing or
`--served-model-name` juggling that a two-pools-in-one-namespace layout needs
applies here.

The router uses the standalone (`epponly`) topology deliberately: its Envoy
routes purely off the destination header via `ORIGINAL_DST`, with no
InferencePool or EndpointSlice membership check, which is what lets it forward
to a gateway in another namespace at all.

## Environment notes

- `kubectl get --raw /api/v1/namespaces/<ns>/services/<svc>:9090/proxy/metrics`
  reads EPP metrics through the API server -- no port-forward, no free local
  port. `tools/analyze_split.py` uses this.
- On a cold Kind cluster the `llm-d-benchmark` image pull can exceed the 6-minute
  data-access wait. Side-load images with `kind load` first (step 1).

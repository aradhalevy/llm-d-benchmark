#!/usr/bin/env bash
# Deploy the cluster-scoped router EPP in front of leaf stacks.
#
#   ROUTER_NS=mc-router ./deploy_router.sh mc-a mc-b            # smart arm
#   ARM=mc-random.yaml ROUTER_NS=mc-router ./deploy_router.sh mc-a mc-b
#
# Each leaf namespace stands in for a peer cluster. Envoy reaches a peer via
# ORIGINAL_DST off the x-gateway-destination-endpoint header, which is parsed as
# an IP:port -- so clusters.yaml carries ClusterIPs, never DNS names.
set -euo pipefail

ROUTER_NS="${ROUTER_NS:-mc-router}"
RELEASE="${RELEASE:-mc-router}"
ARM="${ARM:-mc-smart.yaml}"
CHART_VERSION="${CHART_VERSION:-v0.9.0}"
# Must match the volume's configMap name in router/values.yaml.
CLUSTERS_CM="${CLUSTERS_CM:-mc-clusters}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ $# -ge 2 ]] || { echo "usage: $0 <leaf-ns> <leaf-ns> [...]" >&2; exit 1; }

svc_ip() { kubectl -n "$1" get svc "$2" -o jsonpath='{.spec.clusterIP}'; }

clusters="endpoints:"
for ns in "$@"; do
  gw=$(kubectl -n "$ns" get svc -l gateway.networking.k8s.io/gateway-name \
         -o jsonpath='{.items[0].metadata.name}')
  # Prefer the :80 listener; fall back to the first port for gateways on a
  # custom listenerPort rather than silently assuming 80.
  gw_port=$(kubectl -n "$ns" get svc "$gw" \
         -o jsonpath='{.spec.ports[?(@.port==80)].port}')
  [[ -n $gw_port ]] || gw_port=$(kubectl -n "$ns" get svc "$gw" \
         -o jsonpath='{.spec.ports[0].port}')
  # The EPP Service is the metrics host; the gateway is the routable address.
  epp=$(kubectl -n "$ns" get svc -o name | grep -- '-router-epp$' | head -1 | cut -d/ -f2)
  [[ -n $gw && -n $epp ]] || { echo "$ns: gateway or EPP service not found" >&2; exit 1; }

  clusters+="
  - name: ${ns}
    address: $(svc_ip "$ns" "$gw")
    port: \"${gw_port}\"
    labels:
      metricsAddress: $(svc_ip "$ns" "$epp")
      metricsPort: \"9090\""
done

kubectl get ns "$ROUTER_NS" >/dev/null 2>&1 || kubectl create ns "$ROUTER_NS"
kubectl -n "$ROUTER_NS" create configmap "$CLUSTERS_CM" \
  --from-literal=clusters.yaml="$clusters" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "$clusters"

helm upgrade --install "$RELEASE" \
  oci://ghcr.io/llm-d/charts/llm-d-router-standalone --version "$CHART_VERSION" \
  -n "$ROUTER_NS" -f "$HERE/../router/values.yaml" \
  --set router.epp.pluginsConfigFile="$ARM" \
  --set router.epp.volumes[0].name=clusters \
  --set router.epp.volumes[0].configMap.name="$CLUSTERS_CM"

# Helm does not restart the pod when only the clusters ConfigMap changed.
kubectl -n "$ROUTER_NS" rollout restart "deploy/${RELEASE}-epp"
kubectl -n "$ROUTER_NS" rollout status "deploy/${RELEASE}-epp" --timeout=300s

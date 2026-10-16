# Update Speculator layer IDs on the shared PVC

Run this after training finishes and before evaluation. Training can rewrite the model configs.
It updates only `eagle_aux_hidden_state_layer_ids` to `[2, 14, 25]` in the checkpoint-best and
final model configs in the `efazal/shared` PVC. The checkpoint-best directory currently points to
`speculator-output/0`.

```bash
set -euo pipefail

NAMESPACE=efazal
POD="speculator-config-update-$(date +%s)"
IMAGE="quay.io/opendatahub/odh-th-torch-cuda-py312:odh-3.6-ea.1@sha256:9ad2d72ebe892dffd3554eeb81e2a47c9fcf0d97f89ed261d48f4e8bbc367b4b"

cleanup() {
  oc delete pod "$POD" -n "$NAMESPACE" --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

oc apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 300
  containers:
    - name: config-update
      image: ${IMAGE}
      command: ["python3", "-c", "import time; time.sleep(300)"]
      volumeMounts:
        - name: shared
          mountPath: /mnt/persistent
  volumes:
    - name: shared
      persistentVolumeClaim:
        claimName: shared
YAML

oc wait --for=condition=Ready "pod/$POD" -n "$NAMESPACE" --timeout=120s

oc exec -i -n "$NAMESPACE" "$POD" -- python3 - <<'PY'
import json
from pathlib import Path

layer_ids = [2, 14, 25]
config_paths = [
    Path("/mnt/persistent/speculator-output/checkpoint_best/config.json"),
    Path("/mnt/persistent/final_model/config.json"),
]

configs = []
for path in config_paths:
    if not path.is_file():
        raise FileNotFoundError(path)
    configs.append((path, json.loads(path.read_text())))

for path, config in configs:
    previous = config.get("eagle_aux_hidden_state_layer_ids")
    config["eagle_aux_hidden_state_layer_ids"] = layer_ids
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Updated {path}: {previous} -> {layer_ids}")
PY
```

The temporary pod is deleted automatically when the script exits. This does not rerun the pipeline.

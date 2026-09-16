# Speculative Decoding Serving ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Prerequisites

By default the component returns an internal cluster URL
(`http://<endpoint>-predictor.<namespace>.svc.cluster.local`), which is usable from:

- RHOAI Jupyter workbenches and other pods in the cluster
- Downstream KFP pipeline components in the same run
- Your laptop via `oc port-forward svc/<endpoint>-predictor 8080:80 -n <namespace>`

If you set `enable_external_route=True`, the component creates an OpenShift Route
to expose the endpoint outside the cluster. This requires a one-time admin setup
because the KFP pipeline runner service account (`pipeline-runner-dspa`) does not
have permission to create Routes by default:

```bash
NAMESPACE=<your-rhoai-project-namespace>
sed "s/<NAMESPACE>/$NAMESPACE/g" \
  components/deployment/speculative_decoding_serving/rbac.yaml | oc apply -f -
```

This creates a `Role` and `RoleBinding` granting `pipeline-runner-dspa` permission
to manage Routes in your namespace. It is a one-time step — no need to repeat it
for subsequent pipeline runs. If you skip it and run with `enable_external_route=True`,
the component fails with an error message containing the exact command above.

## Overview 🧾

Serve a verifier and Eagle3 draft model through one vLLM endpoint.

The verifier and draft directories must already exist on ``model_cache_pvc``. KServe mounts the PVC root at ``/mnt/models``. The draft directory should contain the speculator checkpoint's ``config.json`` and weights. vLLM receives an Eagle3 ``--speculative-config`` that points to the draft path
under that root; the verifier is the primary ``--model``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `namespace` | `str` | `None` | Namespace in which to create the KServe resources. |
| `endpoint_name` | `str` | `None` | DNS-compatible InferenceService name. |
| `model_cache_pvc` | `str` | `None` | PVC containing both model directories. |
| `verifier_model_dir` | `str` | `None` | Relative PVC directory containing the verifier model. |
| `draft_model_dir` | `str` | `None` | Relative PVC directory containing the draft checkpoint. |
| `runtime_image` | `str` | `registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:094db84a1da5e8a575d0c9eade114fa30f4a2061064a338e3e032f3578f8082a` | vLLM CUDA image supporting Eagle3 speculative decoding. |
| `serving_runtime_name` | `str` | `""` | ServingRuntime name; defaults to ``<endpoint>-runtime``. |
| `hardware_profile_name` | `str` | `gpu-profile` | RHOAI HardwareProfile name, or empty to skip lookup. |
| `hardware_profile_namespace` | `str` | `redhat-ods-applications` | Namespace containing the HardwareProfile. |
| `min_replicas` | `int` | `1` | Minimum number of predictor replicas. |
| `max_replicas` | `int` | `1` | Maximum number of predictor replicas. |
| `gpu_count` | `int` | `1` | GPUs per predictor; also used as verifier vLLM tensor parallel size. |
| `draft_tensor_parallel_size` | `int` | `1` | vLLM tensor parallel size for the draft model. |
| `max_model_len` | `int` | `4096` | Maximum verifier context length. |
| `num_speculative_tokens` | `int` | `3` | Number of draft tokens proposed per step. |
| `cpu_requests` | `str` | `2` | Predictor CPU request. |
| `memory_requests` | `str` | `8Gi` | Predictor memory request. |
| `cpu_limits` | `str` | `2` | Predictor CPU limit. |
| `memory_limits` | `str` | `8Gi` | Predictor memory limit. |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory available to vLLM. |
| `max_num_seqs` | `int` | `16` | Maximum concurrent sequences in the vLLM scheduler. |
| `trust_remote_code` | `bool` | `True` | Pass ``--trust-remote-code`` for the draft checkpoint. |
| `enable_auth` | `bool` | `False` | Enable RHOAI authentication for the exposed endpoint. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The OpenAI-compatible ``/v1`` endpoint URL. |

## Metadata 🗂️

- **Name**: speculative_decoding_serving
- **Description**: Serve a verifier model and an Eagle3 draft model together with vLLM speculative decoding through a RHOAI KServe InferenceService.

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: OpenShift AI (KServe), Version: >=2.10.0
    - Name: vLLM ServingRuntime, Version: >=0.24.0
- **Tags**:
  - deployment
  - model_serving
  - speculative_decoding
  - vllm
  - kserve
- **Last Verified**: 2026-09-14 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - szaher
    - kryanbeane
    - CathalOConnorRH
  - Reviewers:
    - szaher
    - kryanbeane
    - CathalOConnorRH

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)

<!-- custom-content -->

## Serving Configuration

The draft directory must contain an inference-ready Eagle3 checkpoint, not an
unmodified training checkpoint. For Qwen3 Eagle3 checkpoints, this includes
the expected three auxiliary hidden-state layers and a `target_hidden_size` in
`config.json` matching the verifier hidden size. Export or prepare the
checkpoint before invoking this component; the component does not transform
training checkpoints.

The component mounts the PVC root at `/mnt/models`, then starts vLLM with paths
derived from the two directory inputs. For example, with
`verifier_model_dir=models/qwen3-0.6b` and
`draft_model_dir=speculator/run-01/checkpoint_best`, the configuration is
equivalent to:

```json
{
  "method": "eagle3",
  "model": "/mnt/models/speculator/run-01/checkpoint_best",
  "draft_tensor_parallel_size": 1,
  "num_speculative_tokens": 3
}
```

The draft checkpoint's `config.json` contains the Eagle3 algorithm, layer IDs, proposal defaults, and verifier
metadata such as `speculators_config.verifier.name_or_path`. That metadata identifies the verifier used during
training; it is not a filesystem mount path. The serving component supplies the actual verifier PVC directory as
vLLM's primary model. The runtime image must contain vLLM `>=0.24.0` (or a compatible vendor build) with
Qwen3 Eagle3 support; the `runtime_image` input is the place to provide that RHOAI runtime image.

When `enable_auth` is `True`, RHOAI protects endpoint requests with its platform authentication and authorization
layer. Clients then need a valid bearer token and permission to access the namespace. When it is `False`, callers
that can reach the endpoint can invoke the OpenAI-compatible `/v1` API without an RHOAI user token.

This endpoint authentication is separate from Kubernetes authentication used by the component: the pipeline task
uses its in-cluster ServiceAccount to create and watch the `ServingRuntime` and `InferenceService` resources. The
predictor itself has `automountServiceAccountToken` disabled and does not use that ServiceAccount to authenticate
model requests.

The vLLM reference is [EAGLE Draft Models](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/).

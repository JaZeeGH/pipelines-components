# Speculator Data Only Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract verifier hidden states using managed or external vLLM.

Managed mode starts a vLLM sidecar. When ``download_model`` is false, it uses ``verifier_model_pvc`` as the cached verifier path; when true, it downloads ``verifier_model`` to the PVC. Remote mode calls an existing vLLM endpoint and uses ``verifier_model_pvc`` as the model path visible to that
service. Hidden states are written below ``output_dir`` on the configured storage.

For example, when ``persistent_pvc="shared"``, use either ``verifier_model_pvc="models/Qwen--Qwen2.5-1.5B-Instruct"`` or ``verifier_model_pvc="pvc://shared/models/Qwen--Qwen2.5-1.5B-Instruct"``. Likewise, ``output_dir="speculator-output"`` is equivalent to
``output_dir="pvc://shared/speculator-output"``, and ``hidden_states_path="speculator-output/hidden_states"`` is equivalent to ``hidden_states_path="pvc://shared/speculator-output/hidden_states"``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `persistent_pvc` | `str` | `None` | RWX PVC backing relative model, output, dataset, and hidden-state paths. |
| `download_model` | `bool` | `False` | Download ``verifier_model`` to the PVC when True; otherwise use the cached ``verifier_model_pvc`` path in managed mode. |
| `vllm_source` | `str` | `managed` | Choose ``managed`` sidecar or ``remote`` external vLLM. |
| `verifier_model` | `str` | `""` | Hugging Face verifier ID to download; leave empty to reuse the PVC model. |
| `verifier_model_pvc` | `str` | `models/Qwen--Qwen2.5-1.5B-Instruct` | Verifier path such as ``models/Qwen--Qwen2.5-1.5B-Instruct`` or ``pvc://<claim>/models/Qwen--Qwen2.5-1.5B-Instruct``. |
| `output_dir` | `str` | `speculator-output` | Output directory such as ``speculator-output`` or ``pvc://<claim>/speculator-output``. |
| `dataset_name` | `str` | `ultrachat` | Built-in/Hugging Face dataset reference, local JSON/JSONL path such as ``datasets/train.jsonl``, or ``pvc://<claim>/datasets/train.jsonl``. |
| `hidden_states_path` | `str` | `speculator-output/hidden_states` | Hidden-state path such as ``speculator-output/hidden_states`` or ``pvc://<claim>/speculator-output/hidden_states``. External vLLM must be able to access the same path. |
| `vllm_endpoint` | `str` | `""` | External vLLM URL; required when source is ``remote``. |
| `total_seq_len` | `int` | `2048` | Maximum preprocessing and extraction sequence length. |
| `speculator_type` | `str` | `eagle3` | Draft type; currently ``eagle3``. |
| `target_layer_ids` | `str` | `""` | Four layer IDs for a PVC verifier; empty enables SDK auto-selection for HF models. |
| `data_extraction_max_samples` | `int` | `16` | Samples to process; zero means all. |
| `data_extraction_regenerate_responses` | `bool` | `False` | Regenerate built-in dataset responses with the verifier. |
| `data_extraction_concurrency` | `int` | `1` | Concurrent requests; use ``1`` on shared/NFS PVCs. |
| `vllm_resource_gpu` | `int` | `1` | GPUs for managed vLLM. |
| `vllm_resource_memory` | `str` | `96Gi` | Memory for managed vLLM. |
| `vllm_gpu_memory_utilization` | `float` | `0.9` | vLLM GPU memory fraction. |
| `vllm_readiness_timeout_minutes` | `int` | `60` | vLLM startup timeout. |
| `enable_progression_tracking` | `bool` | `True` | Enable progress reporting. |
| `metrics_port` | `int` | `28080` | Progress metrics port. |
| `metrics_poll_interval_seconds` | `int` | `30` | Progress polling interval. |
| `packages_to_install` | `str` | `""` | Comma-separated TrainJob packages. |
| `pip_index_urls` | `str` | `""` | Comma-separated TrainJob package indexes. |
| `training_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` | Comma-separated TrainJob environment overrides. |
| `training_runtime` | `str` | `vllm-extract-cuda` | ClusterTrainingRuntime for extraction. |

## Metadata 🗂️

- **Name**: speculator_data_only_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - speculator
  - speculative_decoding
  - pipeline
- **Last Verified**: 2026-09-08 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina
    - JaZeeGH
    - Sridhar1030
  - Reviewers:
    - ChughShilpa
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)

# Speculator Data Only Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract verifier hidden states using managed or remote vLLM.

Managed mode runs vLLM in the Trainer job. When ``download_model`` is true, ``verifier_model`` is downloaded to ``persistent_pvc``; otherwise, managed mode reuses ``verifier_model_pvc``. Remote mode uses ``vllm_endpoint`` and requires the verifier and hidden-state PVC paths to be accessible to the
external service. Managed mode writes hidden states under ``output_dir/hidden_states``; remote mode writes them to ``hidden_states_path``.

Relative storage paths are resolved against ``persistent_pvc``. For example, with ``persistent_pvc="shared"``, ``output_dir="speculator-output"`` resolves to ``pvc://shared/speculator-output``. ``speculator-output`` is the default directory name and can be changed.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `persistent_pvc` | `str` | `None` | Existing PVC mounted at ``/mnt/persistent`` for the verifier, local datasets, and outputs. |
| `download_model` | `bool` | `False` | Download ``verifier_model`` to the PVC. Remote mode uses ``verifier_model_pvc``, which must point to the downloaded or cached model. |
| `vllm_source` | `str` | `managed` | Use ``managed`` for a vLLM sidecar in the Trainer job or ``remote`` for an external vLLM service. |
| `verifier_model` | `str` | `""` | Hugging Face Hub model ID to download when ``download_model`` is true; leave empty when reusing ``verifier_model_pvc``. |
| `verifier_model_pvc` | `str` | `models/Qwen--Qwen2.5-1.5B-Instruct` | Path to the cached or downloaded verifier on the PVC; use a relative path or a ``pvc://`` URI. |
| `output_dir` | `str` | `speculator-output` | Trainer output directory on the PVC. ``speculator-output`` is the default; choose another relative path or a URI for that PVC. |
| `dataset_name` | `str` | `ultrachat` | Use ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered), ``ultrachat`` (HuggingFaceH4/ultrachat_200k; default), or ``gsm8k`` (openai/gsm8k). For custom data, use an existing JSON/JSONL file on the PVC (relative path or ``pvc://<claim>/<path>``); arbitrary Hugging Face dataset IDs are not supported. |
| `hidden_states_path` | `str` | `speculator-output/hidden_states` | Remote-mode hidden-state output on the PVC shared with vLLM; managed mode writes to ``<output_dir>/hidden_states``. |
| `vllm_endpoint` | `str` | `""` | URL of the external vLLM service; required for ``remote`` and blank for ``managed``. |
| `total_seq_len` | `int` | `2048` | Maximum sequence length for dataset preprocessing and vLLM extraction; must be positive. |
| `speculator_type` | `str` | `eagle3` | Draft architecture; ``eagle3`` is currently the only supported type. |
| `target_layer_ids` | `str` | `""` | Exactly four comma-separated verifier layer indices; required for the default PVC-backed verifier and remote vLLM. The SDK can infer them only from a Hugging Face model ID passed directly. |
| `data_extraction_max_samples` | `int` | `16` | Maximum samples to process; ``0`` processes the full dataset. |
| `data_extraction_regenerate_responses` | `bool` | `False` | Regenerate responses with the verifier; supported only for ``magpie``, ``ultrachat``, and ``gsm8k``. |
| `data_extraction_concurrency` | `int` | `1` | Number of concurrent vLLM requests for hidden-state extraction; use ``1`` on shared or NFS storage. |
| `vllm_resource_gpu` | `int` | `1` | GPUs requested by managed vLLM; the SDK currently supports exactly ``1``. |
| `vllm_resource_memory` | `str` | `96Gi` | Memory requested by the managed vLLM sidecar. |
| `vllm_gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory available to vLLM, in the range ``(0, 1]``. |
| `vllm_readiness_timeout_minutes` | `int` | `60` | Maximum wait for the managed or remote vLLM endpoint to become ready, in minutes; minimum ``1``. |
| `enable_progression_tracking` | `bool` | `True` | Enable Speculator extraction progress reporting. |
| `metrics_port` | `int` | `28080` | HTTP port for extraction metrics; valid range is ``1024``-``65535``. |
| `metrics_poll_interval_seconds` | `int` | `30` | Seconds between metrics polls; valid range is ``5``-``300``. |
| `packages_to_install` | `str` | `""` | Comma-separated Python packages to install in the Trainer job. |
| `pip_index_urls` | `str` | `""` | Comma-separated package-index URLs used for Trainer job installs. |
| `training_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` | Comma-separated ``KEY=VALUE`` environment overrides for the Trainer job. |
| `training_runtime` | `str` | `vllm-extract-cuda` | Name of the ClusterTrainingRuntime used by the Trainer job. |

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

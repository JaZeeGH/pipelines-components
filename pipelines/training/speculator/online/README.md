# Speculator Online Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train, evaluate, and serve an Eagle3 speculative decoding model.

Serving requires the namespace and a verified vLLM runtime image pull spec.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `persistent_pvc` | `str` | `None` |  |
| `serving_namespace` | `str` | `None` |  |
| `serving_runtime_image` | `str` | `None` |  |
| `evaluation_dataset_uri` | `str` | `RedHatAI/speculator_benchmarks` |  |
| `serving_endpoint_name` | `str` | `speculator-online` |  |
| `verifier_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` |  |
| `verifier_model_pvc` | `str` | `models/Qwen--Qwen2.5-1.5B-Instruct` |  |
| `output_dir` | `str` | `speculator-output` |  |
| `dataset_name` | `str` | `ultrachat` |  |
| `total_seq_len` | `int` | `2048` |  |
| `speculator_type` | `str` | `eagle3` |  |
| `target_layer_ids` | `str` | `""` |  |
| `data_extraction_max_samples` | `int` | `16` |  |
| `data_extraction_regenerate_responses` | `bool` | `False` |  |
| `data_extraction_concurrency` | `int` | `4` |  |
| `training_epochs` | `int` | `1` |  |
| `training_lr` | `float` | `0.0001` |  |
| `training_draft_vocab_size` | `int` | `0` |  |
| `training_num_layers` | `int` | `1` |  |
| `training_ttt_steps` | `int` | `1` |  |
| `training_norm_before_residual` | `bool` | `True` |  |
| `training_norm_before_fc` | `bool` | `False` |  |
| `training_embed_requires_grad` | `bool` | `False` |  |
| `training_hidden_states_dtype` | `str` | `bfloat16` |  |
| `training_scheduler_type` | `str` | `linear` |  |
| `training_scheduler_warmup_steps` | `int` | `0` |  |
| `training_scheduler_total_steps` | `int` | `0` |  |
| `training_scheduler_num_cosine_cycles` | `float` | `0.5` |  |
| `training_checkpoint_freq` | `float` | `1.0` |  |
| `training_save_best` | `bool` | `False` |  |
| `training_log_freq` | `int` | `1` |  |
| `training_resume_from_checkpoint` | `bool` | `False` |  |
| `training_from_pretrained` | `str` | `""` |  |
| `training_resource_cpu` | `str` | `4` |  |
| `training_resource_gpu` | `int` | `1` |  |
| `training_resource_memory` | `str` | `64Gi` |  |
| `vllm_resource_gpu` | `int` | `1` |  |
| `vllm_resource_memory` | `str` | `96Gi` |  |
| `vllm_gpu_memory_utilization` | `float` | `0.9` |  |
| `vllm_readiness_timeout_minutes` | `int` | `60` |  |
| `enable_progression_tracking` | `bool` | `True` |  |
| `metrics_port` | `int` | `28080` |  |
| `metrics_poll_interval_seconds` | `int` | `30` |  |
| `packages_to_install` | `str` | `""` |  |
| `pip_index_urls` | `str` | `""` |  |
| `training_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` |  |
| `training_runtime` | `str` | `vllm-extract-cuda` |  |
| `evaluation_max_samples` | `int` | `80` |  |
| `evaluation_max_tokens` | `int` | `256` |  |
| `evaluation_temperature` | `float` | `0.0` |  |
| `evaluation_num_speculative_tokens` | `int` | `5` |  |
| `evaluation_gpu_memory_utilization` | `float` | `0.9` |  |
| `evaluation_startup_timeout_seconds` | `int` | `900` |  |
| `evaluation_sample_seed` | `int` | `42` |  |

## Metadata 🗂️

- **Name**: speculator_online_pipeline
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
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)

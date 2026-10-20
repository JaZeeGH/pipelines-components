# Speculator Train Only Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train, evaluate, register, and serve a draft from existing artifacts.

This pipeline runs Trainer's train-only mode and expects the verifier model, hidden states, and preprocessed Arrow data to already exist on the shared PVC. It does not download the verifier. Relative storage paths are resolved against ``persistent_pvc``. Explicit ``pvc://<claim>/<path>`` paths must
reference that same PVC.

For example, with ``persistent_pvc="shared"``, ``verifier_model_pvc="models/Qwen--Qwen3-0.6B"`` is equivalent to ``pvc://shared/models/Qwen--Qwen3-0.6B``. Likewise, ``output_dir="speculator-output"`` and ``hidden_states_path="speculator-output/hidden_states"`` can be written as
``pvc://shared/speculator-output`` and ``pvc://shared/speculator-output/hidden_states``. The default ``training_data_path`` and ``output_dir`` are both ``speculator-output``, matching the data-only pipeline defaults. If you customize the data-only output directory, update ``training_data_path`` and
``hidden_states_path`` too. The hidden-state default matches the data-only pipeline's default output directory; it is not discovered automatically from a previous pipeline run.

The trained draft is copied to ``pvc://<persistent_pvc>/final_model`` for evaluation, Model Registry, and serving. The resulting KServe endpoint uses cluster-local visibility.

Model Registry setup: In the RHOAI dashboard, open **AI Hub > Models > Model registries** and create or select a registry. Copy its HTTP REST service URL. This pipeline takes the host and port separately: put only the hostname in ``model_registry_address`` and the service port in
``model_registry_port``. For a URL such as ``http://registry-service.registry-namespace.svc:8080/api/model_registry/v1alpha3``, use ``registry-service.registry-namespace.svc`` and ``8080``; omit the scheme and API path. The pipeline ServiceAccount must have a RoleBinding to
``registry-user-<registry-name>`` in the registry namespace. ``model_registry_address`` is required because this pipeline registers the evaluated model before serving it. Use a new ``model_registry_version`` for each trained revision.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `persistent_pvc` | `str` | `None` | Name only of an existing ReadWriteMany PVC mounted at ``/mnt/persistent``; do not include a path or ``pvc://`` prefix. It stores the verifier, hidden states, preprocessed data, and model output. |
| `serving_namespace` | `str` | `None` | OpenShift namespace where KServe creates the endpoint. It must be the PVC's namespace and have the required KServe permissions. |
| `model_registry_address` | `str` | `None` | Required HTTP REST host from AI Hub > Models > Model registries. Enter only the hostname, without scheme or API path. |
| `model_registry_name` | `str` | `None` | Required model name; choose a name unique within the selected registry. |
| `serving_endpoint_name` | `str` | `None` | Required DNS-compatible KServe InferenceService name, unique in ``serving_namespace``. |
| `model_registry_port` | `int` | `8080` | HTTP REST service port copied from the registry endpoint; ``8080`` is the default. |
| `model_registry_version` | `str` | `1.0.0` | Version label for this registration. Use a new version for each trained revision. |
| `serving_runtime_image` | `str` | `registry.redhat.io/rhaii-fast/vllm-cuda-rhel9@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a` | Eagle3-compatible vLLM image pull spec. The default is the pinned Red Hat runtime; override only with a verified digest. |
| `evaluation_dataset_uri` | `str` | `RedHatAI/speculator_benchmarks` | Hugging Face dataset ID. The supported default, ``RedHatAI/speculator_benchmarks``, runs its ``HumanEval`` subset. For local data, provide an existing JSONL file on ``persistent_pvc`` as a relative path or matching ``pvc://`` URI. |
| `verifier_model_pvc` | `str` | `models/Qwen--Qwen3-0.6B` | Required existing verifier directory on ``persistent_pvc``; the default is ``models/Qwen--Qwen3-0.6B``. Use a relative path or ``pvc://<persistent_pvc>/<path>``. Training, evaluation, and serving all read this directory. |
| `output_dir` | `str` | `speculator-output` | Trainer checkpoint/output directory on ``persistent_pvc``. Default: ``speculator-output``. Use a relative path or matching URI. |
| `hidden_states_path` | `str` | `speculator-output/hidden_states` | Existing hidden-state directory from data-only extraction. The default ``speculator-output/hidden_states`` matches the data-only default ``output_dir="speculator-output"``; it is not auto-detected. Override it if the data-only run used a different output directory. Use a relative path or matching ``pvc://`` URI. |
| `training_data_path` | `str` | `speculator-output` | Preprocessed Arrow dataset directory from data-only extraction. Default: ``speculator-output``. Set it to the data-only output path (relative or matching ``pvc://`` URI); an empty value falls back to ``output_dir``. |
| `total_seq_len` | `int` | `2048` | Positive sequence length for preprocessing and training; keep it consistent with the data-only extraction run. |
| `speculator_type` | `str` | `eagle3` | Draft architecture. ``eagle3`` is currently the supported training type. |
| `target_layer_ids` | `str` | `2,14,25,28` | Exactly four comma-separated verifier layer indices. Default ``2,14,25,28`` matches the default 28-layer Qwen3-0.6B model. Use the same four indices as data-only extraction; set matching indices for another verifier. |
| `training_epochs` | `int` | `3` | Number of training epochs; must be positive. |
| `training_lr` | `float` | `0.0001` | Positive learning rate for draft-model training. |
| `training_draft_vocab_size` | `int` | `0` | Draft vocabulary size. ``0`` lets the SDK derive it from the verifier vocabulary. |
| `training_num_layers` | `int` | `1` | Number of layers in the draft model. |
| `training_ttt_steps` | `int` | `1` | Number of test-time-training steps. |
| `training_norm_before_residual` | `bool` | `True` | Apply normalization before the residual connection. |
| `training_norm_before_fc` | `bool` | `False` | Apply normalization before the fully connected layer. |
| `training_embed_requires_grad` | `bool` | `False` | Whether the draft embedding weights are trainable. |
| `training_hidden_states_dtype` | `str` | `bfloat16` | Hidden-state tensor dtype: ``bfloat16``, ``float16``, or ``float32``; it must match the verifier. |
| `training_scheduler_type` | `str` | `linear` | Learning-rate scheduler: ``linear``, ``cosine``, or ``none``. |
| `training_scheduler_warmup_steps` | `int` | `0` | Warmup steps; ``0`` lets the SDK compute 1% of total training steps. |
| `training_scheduler_total_steps` | `int` | `0` | Total scheduler steps; ``0`` lets the SDK calculate the value from epochs and dataset size. |
| `training_scheduler_num_cosine_cycles` | `float` | `0.5` | Cosine cycles; used only when ``training_scheduler_type="cosine"``. |
| `training_checkpoint_freq` | `float` | `1.0` | Checkpoint interval in epochs. |
| `training_save_best` | `bool` | `False` | Save only the best checkpoint by validation loss. |
| `training_log_freq` | `int` | `1` | Training steps between log updates. |
| `training_resume_from_checkpoint` | `bool` | `False` | Resume training from a checkpoint when the runtime supports it. |
| `training_resource_cpu` | `str` | `4` | CPU request for the TrainJob worker, as a Kubernetes quantity. |
| `training_resource_gpu` | `int` | `1` | Number of GPUs requested for the TrainJob worker. |
| `training_resource_memory` | `str` | `64Gi` | Memory request for the TrainJob worker, as a Kubernetes quantity. |
| `enable_progression_tracking` | `bool` | `True` | Enable Trainer progress reporting. |
| `metrics_port` | `int` | `28080` | HTTP port for training metrics; valid range is ``1024``-``65535``. |
| `metrics_poll_interval_seconds` | `int` | `30` | Seconds between metrics polls; valid range is ``5``-``300``. |
| `packages_to_install` | `str` | `""` | Comma-separated Python packages to install in the TrainJob. |
| `pip_index_urls` | `str` | `""` | Comma-separated package-index URLs for TrainJob installs. |
| `training_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` | Comma-separated ``KEY=VALUE`` environment overrides for the TrainJob. |
| `training_runtime` | `str` | `speculator-model-opt-cuda` | Name of the ClusterTrainingRuntime used by the TrainJob. |
| `evaluation_max_samples` | `int` | `80` | Maximum requests per evaluation sweep point; must be positive. |
| `evaluation_max_tokens` | `int` | `256` | Maximum generated tokens per evaluation request; must be positive. |
| `evaluation_temperature` | `float` | `0.0` | Sampling temperature used by GuideLLM; ``0`` is deterministic. |
| `evaluation_num_speculative_tokens` | `int` | `5` | Number of draft tokens proposed per step; also used by serving. |
| `evaluation_gpu_memory_utilization` | `float` | `0.9` | GPU memory fraction for evaluation and serving; range ``(0, 1]``. |
| `evaluation_startup_timeout_seconds` | `int` | `900` | Maximum seconds to wait for the evaluation vLLM server to become ready. |
| `evaluation_sample_seed` | `int` | `42` | Seed for deterministic sampling from a local evaluation dataset. |

## Metadata 🗂️

- **Name**: speculator_train_only_pipeline
- **Description**: Train a Speculator Eagle3 draft from pre-extracted hidden states, benchmark it, register it, and serve it with vLLM.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Model Registry, Version: >=0.3.4
    - Name: OpenShift AI (KServe), Version: >=2.10.0
    - Name: vLLM ServingRuntime, Version: >=0.24.0
    - Name: vLLM, Version: >=0.9.0
    - Name: GuideLLM, Version: ~=0.7.1
- **Tags**:
  - training
  - speculator
  - speculative_decoding
  - pipeline
- **Last Verified**: 2026-10-20 00:00:00+00:00
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

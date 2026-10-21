# Speculator Offline Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract verifier hidden states for speculative decoding training with external vLLM.

Then train and evaluate a draft, optionally register it, and serve it.

This offline pipeline combines data-only extraction and train-only training in a single run. It creates hidden states and processed training data before training, so pre-existing extraction artifacts are not required.

### Required inputs and shared storage

``p01_persistent_pvc`` is an existing ReadWriteMany PVC in the pipeline namespace. It stores the verifier, datasets, extraction outputs, training checkpoints, and final draft, and must also be available in ``p50_serving_namespace``. Its ``p50_serving_namespace`` input must be set for a run. The PVC
is mounted at ``/mnt/persistent``. With ``p01_persistent_pvc="shared"``, ``p08_output_dir="speculator-output"`` resolves to ``pvc://shared/speculator-output``.

The parameter prefixes follow execution order: storage, verifier, extraction, paths, architecture, training, evaluation, registry, and serving. The Registry address ``p46_model_registry_address`` is optional; leave it empty to skip registration. If provided, enter only the hostname and omit the URL
scheme and API path. The pipeline ServiceAccount needs a RoleBinding to ``registry-user-<registry-name>`` in the Registry namespace.

### Verifier model and data extraction

``p02_verifier_model_pvc`` must identify the relative verifier directory on the PVC, and the verifier model must already exist there. This pipeline does not download models from HuggingFace.

The extraction stage sends requests to the external vLLM endpoint in ``p03_vllm_endpoint``; it does not launch a managed vLLM sidecar. The endpoint must be provided and able to access the PVC-backed verifier and hidden states. ``p04_dataset_name`` accepts ``magpie``
(Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered), ``ultrachat`` (HuggingFaceH4/ultrachat_200k), or ``gsm8k`` (openai/gsm8k), and can also identify a JSON/JSONL dataset already on the PVC. Only these built-in datasets and local JSON/JSONL files are supported in offline mode.

Extraction writes hidden states to ``p09_hidden_states_path`` and processed data below ``p08_output_dir``. Training reads processed data from ``p10_training_data_path`` and hidden states from ``p09_hidden_states_path``. Their defaults match the extraction output; if ``p08_output_dir`` changes,
update both paths. ``p13_target_layer_ids`` defaults to empty; supply the verifier's four comma-separated layer IDs. ``p05_data_extraction_max_samples=0`` processes all examples; response regeneration (``p06_data_extraction_regenerate_responses``) applies to built-in datasets, and concurrency
``p07_data_extraction_concurrency=1`` is recommended on shared/NFS PVCs.

Extraction and training use the same ``p35_job_runtime``. The jobs share environment overrides, packages, and package indexes through ``p34_job_envs``, ``p36_packages_to_install``, and ``p37_pip_index_urls``. A ``kubernetes-credentials`` secret with ``KUBERNETES_SERVER_URL`` and
``KUBERNETES_AUTH_TOKEN`` is required by both Trainer stages.

### Training and evaluation

Training runs after extraction. A draft vocabulary size of ``0`` lets Trainer derive it; scheduler warmup and total steps of ``0`` let Trainer calculate them. The completed draft is written to ``pvc://<p01_persistent_pvc>/final_model``. Each run replaces this shared path; use a separate PVC per
revision if earlier Registry versions must continue referencing their original weights.

``p38_evaluation_dataset_uri`` must identify a Hugging Face dataset ID or a JSON/JSONL file on the PVC. Local paths are relative to the PVC mount, and ``pvc://`` URIs must reference ``p01_persistent_pvc``. Avoid leading or trailing whitespace. Evaluation receives only the optional ``hf-token``
secret, not Kubernetes Trainer credentials. ``p39``–``p45`` control the evaluation request limit, generation, sampling, and startup timeout.

### Model Registry and serving

In the RHOAI dashboard, open **AI Hub > Models > Model registries** and create or select a registry. Copy its HTTP REST service URL. Put only the hostname in ``p46_model_registry_address`` and the service port in ``p47_model_registry_port``; omit the scheme and API path. For example, for
``http://registry-service.registry-namespace.svc:8080/api/model_registry/v1alpha3``, use ``registry-service.registry-namespace.svc`` and ``8080``. Set a unique model name in ``p48_model_registry_name`` and use a new ``p49_model_registry_version`` for each revision.

Serving creates a cluster-local KServe InferenceService using the verifier at ``p02_verifier_model_pvc`` and the draft at ``final_model``. ``p50_serving_namespace`` must access the PVC; the pipeline ServiceAccount needs permission to manage the ServingRuntime and InferenceService and read the
predictor Deployment. ``p51_serving_endpoint_name`` must be DNS-compatible. The pinned compatible vLLM CUDA image is the default for ``p52_serving_runtime_image``. ``p11_total_seq_len``, ``p42_evaluation_num_speculative_tokens``, and ``p43_evaluation_gpu_memory_utilization`` apply to evaluation and
serving.

Zero-padded ``pNN_`` prefixes preserve parameter order in the KFP UI.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `p01_persistent_pvc` | `str` | `None` | Required name of the existing ReadWriteMany PVC in the pipeline namespace. It stores the verifier, datasets, extraction outputs, checkpoints, and final draft; KServe must also access it in ``p50_serving_namespace``. The PVC is mounted at ``/mnt/persistent``. |
| `p02_verifier_model_pvc` | `str` | `None` | Required verifier model directory relative to the PVC. The model must already exist at this path; the pipeline does not download verifier models from HuggingFace. |
| `p03_vllm_endpoint` | `str` | `None` | Required URL for the external vLLM service. Set it to an endpoint that can access the verifier and hidden-state paths on ``p01_persistent_pvc``. |
| `p04_dataset_name` | `str` | `None` | Built-in dataset name or an existing JSON/JSONL dataset on the PVC. Supported built-ins are ``magpie``, ``ultrachat``, and ``gsm8k``; only these and local JSON/JSONL files are supported in offline mode. |
| `p05_data_extraction_max_samples` | `int` | `0` | Maximum extraction examples. Set to ``0`` to process the full dataset. |
| `p06_data_extraction_regenerate_responses` | `bool` | `False` | Regenerate assistant responses with the verifier instead of using original responses. Supported only for the built-in datasets. |
| `p07_data_extraction_concurrency` | `int` | `1` | Number of concurrent extraction requests. Use ``1`` on shared or NFS PVCs. |
| `p08_output_dir` | `str` | `speculator-output` | PVC-relative directory for extraction output, processed training data, and checkpoints. The final draft is copied to ``final_model`` at the PVC root. |
| `p09_hidden_states_path` | `str` | `speculator-output/hidden_states` | PVC-relative hidden-state directory written by extraction and read by training. Keep this path aligned with the output directory. |
| `p10_training_data_path` | `str` | `speculator-output` | Directory containing the processed dataset written by extraction. Keep it aligned with ``p08_output_dir`` when changing the defaults. |
| `p11_total_seq_len` | `int` | `2048` | Positive maximum sequence length used for extraction, training, evaluation, and serving. |
| `p12_speculator_type` | `str` | `eagle3` | Draft architecture. The current Trainer and evaluator support ``eagle3``. |
| `p13_target_layer_ids` | `str` | `""` | Four comma-separated verifier layer IDs. It defaults to empty; provide the IDs for the selected verifier. |
| `p14_training_epochs` | `int` | `3` | Number of training epochs. |
| `p15_training_lr` | `float` | `0.0001` | Learning rate for draft-model training. |
| `p16_training_draft_vocab_size` | `int` | `0` | Draft vocabulary size. ``0`` lets the Trainer SDK derive it. |
| `p17_training_num_layers` | `int` | `1` | Number of layers in the draft model. |
| `p18_training_test_time_steps` | `int` | `1` | Number of test-time training steps configured for Speculator. |
| `p19_training_norm_before_residual` | `bool` | `True` | Apply normalization before the residual connection. |
| `p20_training_norm_before_fc` | `bool` | `False` | Apply normalization before the fully connected layer. |
| `p21_training_embed_requires_grad` | `bool` | `False` | Whether draft embedding weights are trainable. |
| `p22_training_hidden_states_dtype` | `str` | `bfloat16` | Hidden-state dtype; it must match the verifier model. |
| `p23_training_scheduler_type` | `str` | `linear` | Learning-rate scheduler: ``linear``, ``cosine``, or ``none``. |
| `p24_training_scheduler_warmup_steps` | `int` | `0` | Warmup steps. ``0`` lets the SDK calculate them. |
| `p25_training_scheduler_total_steps` | `int` | `0` | Total scheduler steps. ``0`` lets the SDK calculate them. |
| `p26_training_scheduler_num_cosine_cycles` | `float` | `0.5` | Number of cosine cycles; ignored by other scheduler types. |
| `p27_training_checkpoint_freq` | `float` | `1.0` | Checkpoint interval in epochs. |
| `p28_training_save_best` | `bool` | `False` | Save the best checkpoint by validation loss. |
| `p29_training_log_freq` | `int` | `1` | Number of training steps between log updates. |
| `p30_training_resume_from_checkpoint` | `bool` | `False` | Resume from a saved checkpoint when supported by the runtime. |
| `p31_training_resource_cpu` | `str` | `4` | CPU request for the training worker. |
| `p32_training_resource_gpu` | `int` | `1` | Number of GPUs requested for the training worker. |
| `p33_training_resource_memory` | `str` | `64Gi` | Memory request for the training worker. |
| `p34_job_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` | Comma-separated ``KEY=VALUE`` environment overrides shared by extraction and training jobs. |
| `p35_job_runtime` | `str` | `speculator-model-opt-cuda` | ClusterTrainingRuntime shared by hidden-state extraction and draft training. |
| `p36_packages_to_install` | `str` | `""` | Comma-separated Python packages installed in extraction and training jobs. |
| `p37_pip_index_urls` | `str` | `""` | Comma-separated package-index URLs used when installing job packages. |
| `p38_evaluation_dataset_uri` | `str` | `""` | Hugging Face dataset ID or JSON/JSONL path; must be set; a ``pvc://`` URI must reference ``p01_persistent_pvc``. |
| `p39_evaluation_max_samples` | `int` | `80` | Maximum GuideLLM requests per evaluation sweep point; must be positive. |
| `p40_evaluation_max_tokens` | `int` | `256` | Maximum tokens generated per evaluation request; must be positive. |
| `p41_evaluation_temperature` | `float` | `0.0` | Sampling temperature passed to GuideLLM. ``0`` is deterministic. |
| `p42_evaluation_num_speculative_tokens` | `int` | `5` | Draft tokens proposed per step in evaluation and serving. |
| `p43_evaluation_gpu_memory_utilization` | `float` | `0.9` | GPU memory fraction for evaluation and serving, greater than ``0`` and at most ``1.0``. |
| `p44_evaluation_startup_timeout_seconds` | `int` | `900` | Maximum wait for the evaluation vLLM server to become ready. |
| `p45_evaluation_sample_seed` | `int` | `42` | Seed used for deterministic sampling of local evaluation datasets. |
| `p46_model_registry_address` | `str` | `""` | Model Registry hostname. Enter only the host, not the scheme or API path. Its default is empty, which skips registration. |
| `p47_model_registry_port` | `int` | `8080` | HTTP REST service port copied from the Model Registry URL; ``8080`` is common. |
| `p48_model_registry_name` | `str` | `speculator-offline-draft` | Name assigned to the registered draft model; use a unique name in the registry. |
| `p49_model_registry_version` | `str` | `1.0.0` | Version label for this registration. Use a new value for each revision. |
| `p50_serving_namespace` | `str` | `""` | KServe namespace; must be set to access the PVC. |
| `p51_serving_endpoint_name` | `str` | `speculator-offline` | DNS-compatible name for the KServe InferenceService. |
| `p52_serving_runtime_image` | `str` | `registry.redhat.io/rhaii-fast/vllm-cuda-rhel9@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a` | Pinned compatible vLLM CUDA image pull spec used by KServe; override only with a compatible image. |

## Metadata 🗂️

- **Name**: speculator_offline_pipeline
- **Description**: Extract verifier hidden states for speculative decoding training with external vLLM, train and evaluate the draft, optionally register it, then serve it.
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
- **Last Verified**: 2026-10-21 00:00:00+00:00
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

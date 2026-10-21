"""Speculator offline extraction, training, evaluation, and deployment pipeline."""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.download_model import download_model as download_model_component
from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.deployment.speculative_decoding_serving import speculative_decoding_serving
from components.evaluation.speculator import evaluate_speculator
from components.training.finetuning.speculator import extract_speculator, train_speculator_mode

PVC_SIZE = "200Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "speculator-offline-pipeline"


def _attach_hf_token(task) -> None:
    """Attach the optional Hugging Face token to a task."""
    kfp.kubernetes.use_secret_as_env(
        task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )


def _attach_trainer_credentials(task) -> None:
    """Attach Kubernetes and optional Hugging Face credentials to a Trainer task."""
    task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(task, "IfNotPresent")
    kfp.kubernetes.use_secret_as_env(
        task,
        secret_name="kubernetes-credentials",
        secret_key_to_env={
            "KUBERNETES_SERVER_URL": "KUBERNETES_SERVER_URL",
            "KUBERNETES_AUTH_TOKEN": "KUBERNETES_AUTH_TOKEN",
        },
        optional=False,
    )
    _attach_hf_token(task)


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "Extract verifier hidden states for speculative decoding training with managed vLLM, train and evaluate "
        "the draft, optionally register it, then serve it."
    ),
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size=PVC_SIZE,
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "accessModes": PVC_ACCESS_MODES,
                    "storageClassName": PVC_STORAGE_CLASS,
                }
            ),
        ),
    ),
)
def speculator_offline_pipeline(
    # Required inputs & storage
    p01_persistent_pvc: str,
    # Verifier model
    p02_verifier_model: str = "",
    p03_verifier_model_pvc: str = "",
    p04_download_model: bool = False,
    # Data extraction configuration
    p05_dataset_name: str = "ultrachat",
    p06_dataset_download_enabled: bool = False,
    p07_downloaded_dataset_uri: str = "",
    p08_data_extraction_max_samples: int = 16,
    p09_data_extraction_regenerate_responses: bool = False,
    p10_data_extraction_concurrency: int = 1,
    p11_vllm_resource_gpu: int = 1,
    p12_vllm_resource_memory: str = "96Gi",
    p13_vllm_gpu_memory_utilization: float = 0.9,
    p14_vllm_readiness_timeout_minutes: int = 60,
    p15_extraction_runtime: str = "vllm-extract-cuda",
    # Storage paths
    p16_output_dir: str = "speculator-output",
    p17_hidden_states_path: str = "speculator-output/hidden_states",
    p18_training_data_path: str = "speculator-output",
    # Model architecture
    p19_total_seq_len: int = 2048,
    p20_speculator_type: str = "eagle3",
    p21_target_layer_ids: str = "",
    # Training configuration
    p22_training_epochs: int = 3,
    p23_training_lr: float = 0.0001,
    p24_training_draft_vocab_size: int = 0,
    p25_training_num_layers: int = 1,
    p26_training_test_time_steps: int = 1,
    p27_training_norm_before_residual: bool = True,
    p28_training_norm_before_fc: bool = False,
    p29_training_embed_requires_grad: bool = False,
    p30_training_hidden_states_dtype: str = "bfloat16",
    p31_training_scheduler_type: str = "linear",
    p32_training_scheduler_warmup_steps: int = 0,
    p33_training_scheduler_total_steps: int = 0,
    p34_training_scheduler_num_cosine_cycles: float = 0.5,
    p35_training_checkpoint_freq: float = 1.0,
    p36_training_save_best: bool = False,
    p37_training_log_freq: int = 1,
    p38_training_resume_from_checkpoint: bool = False,
    p39_training_resource_cpu: str = "4",
    p40_training_resource_gpu: int = 1,
    p41_training_resource_memory: str = "64Gi",
    p42_job_envs: str = "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1",
    p43_job_runtime: str = "speculator-model-opt-cuda",
    p44_packages_to_install: str = "",
    p45_pip_index_urls: str = "",
    # Evaluation configuration
    p46_evaluation_dataset_uri: str = "",
    p47_evaluation_max_samples: int = 80,
    p48_evaluation_max_tokens: int = 256,
    p49_evaluation_temperature: float = 0.0,
    p50_evaluation_num_speculative_tokens: int = 5,
    p51_evaluation_gpu_memory_utilization: float = 0.9,
    p52_evaluation_startup_timeout_seconds: int = 900,
    p53_evaluation_sample_seed: int = 42,
    # Model registry
    p54_model_registry_address: str = "",
    p55_model_registry_port: int = 8080,
    p56_model_registry_name: str = "speculator-offline-draft",
    p57_model_registry_version: str = "1.0.0",
    # Serving deployment
    p58_serving_namespace: str = "",
    p59_serving_endpoint_name: str = "speculator-offline",
    p60_serving_runtime_image: str = (
        "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9"
        "@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a"
    ),
):
    """Extract verifier hidden states for speculative decoding training with managed vLLM.

    Then train and evaluate a draft, optionally register it, and serve it.

    This offline pipeline combines data-only extraction and train-only training in a
    single run. It creates hidden states and processed training data before training,
    so pre-existing extraction artifacts are not required.

    ### Required inputs and shared storage

    ``p01_persistent_pvc`` is an existing ReadWriteMany PVC in the pipeline namespace.
    It stores the verifier, datasets, extraction outputs, training checkpoints, and
    final draft, and must also be available in ``p58_serving_namespace``. Its
    ``p58_serving_namespace`` input must be set for a run. The
    PVC is mounted at ``/mnt/persistent``. With ``p01_persistent_pvc="shared"``,
    ``p16_output_dir="speculator-output"`` resolves to
    ``pvc://shared/speculator-output``.

    The parameter prefixes follow execution order: storage, verifier, extraction,
    paths, architecture, training, evaluation, registry, and serving. The Registry
    address ``p54_model_registry_address`` is optional; leave it empty to skip
    registration. If provided, enter only the hostname and omit the URL scheme and
    API path. The pipeline ServiceAccount needs a RoleBinding to
    ``registry-user-<registry-name>`` in the Registry namespace.

    ### Verifier model and data extraction

    ``p03_verifier_model_pvc`` must identify the relative
    verifier directory on the PVC. When ``p04_download_model`` is false, the model
    must already exist there. When true,
    ``p02_verifier_model`` is downloaded under ``models/`` in a directory named after
    the model ID with slashes replaced by ``--``; set ``p03_verifier_model_pvc`` to
    that same path. Configure the optional ``hf-token`` secret for gated models or
    datasets.

    The extraction stage uses managed vLLM in the Trainer job. ``p05_dataset_name``
    accepts ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered), ``ultrachat``
    (HuggingFaceH4/ultrachat_200k), or ``gsm8k`` (openai/gsm8k), and can also identify
    a JSON/JSONL dataset already on the PVC. Arbitrary Hugging Face dataset IDs are
    not accepted through this parameter. To forward a custom dataset URI to Trainer,
    enable ``p06_dataset_download_enabled`` and set ``p07_downloaded_dataset_uri``;
    the URI is required when that option is enabled, and PVC URIs must reference
    ``p01_persistent_pvc``.

    Managed extraction writes hidden states below
    ``p16_output_dir/hidden_states``. Training reads processed data from
    ``p18_training_data_path`` and hidden states from ``p17_hidden_states_path``.
    Their defaults match the extraction output; if ``p16_output_dir`` changes, update
    both paths. ``p21_target_layer_ids`` defaults to empty; supply the verifier's
    four comma-separated layer IDs. ``p08_data_extraction_max_samples=0`` processes all
    examples; response regeneration applies to built-in datasets, and concurrency
    ``p10_data_extraction_concurrency=1`` is recommended on shared/NFS PVCs.

    Extraction uses ``p15_extraction_runtime`` and training uses
    ``p43_job_runtime``. The jobs share environment overrides, packages, and package
    indexes through ``p42_job_envs``, ``p44_packages_to_install``, and
    ``p45_pip_index_urls``. A ``kubernetes-credentials`` secret with
    ``KUBERNETES_SERVER_URL`` and ``KUBERNETES_AUTH_TOKEN`` is required by both
    Trainer stages.

    ### Training and evaluation

    Training runs after extraction. A draft vocabulary size of ``0`` lets Trainer
    derive it; scheduler warmup and total steps of ``0`` let Trainer calculate them.
    The completed draft is written to ``pvc://<p01_persistent_pvc>/final_model``.
    Each run replaces this shared path; use a separate PVC per revision if earlier
    Registry versions must continue referencing their original weights.

    ``p46_evaluation_dataset_uri`` must identify a Hugging Face
    dataset ID or a JSON/JSONL file on the PVC. Local paths are relative to the PVC mount, and
    ``pvc://`` URIs must reference ``p01_persistent_pvc``. Avoid leading or trailing
    whitespace. Evaluation receives only the optional ``hf-token`` secret, not
    Kubernetes Trainer credentials.
    ``p47``–``p53`` control the evaluation request limit, generation, sampling, and
    startup timeout.

    ### Model Registry and serving

    In the RHOAI dashboard, open **AI Hub > Models > Model registries** and create or
    select a registry. Copy its HTTP REST service URL. Put only the hostname in
    ``p54_model_registry_address`` and the service port in ``p55_model_registry_port``;
    omit the scheme and API path. For example, for
    ``http://registry-service.registry-namespace.svc:8080/api/model_registry/v1alpha3``,
    use ``registry-service.registry-namespace.svc`` and ``8080``. Set a unique model
    name in ``p56_model_registry_name`` and use a new ``p57_model_registry_version``
    for each revision.

    Serving creates a cluster-local KServe InferenceService using the verifier at
    ``p03_verifier_model_pvc`` and the draft at ``final_model``. ``p58_serving_namespace``
    must access the PVC; the pipeline ServiceAccount needs permission to manage the
    ServingRuntime and InferenceService and read the predictor Deployment.
    ``p59_serving_endpoint_name`` must be DNS-compatible. The pinned compatible vLLM
    CUDA image is the default for ``p60_serving_runtime_image``.
    ``p19_total_seq_len``, ``p50_evaluation_num_speculative_tokens``, and
    ``p51_evaluation_gpu_memory_utilization`` apply to evaluation and serving.

    Zero-padded ``pNN_`` prefixes preserve parameter order in the KFP UI.

    Args:
        p01_persistent_pvc: Required name of the existing ReadWriteMany PVC in the pipeline namespace. It stores
            the verifier, datasets, extraction outputs, checkpoints, and final draft; KServe must also access it in
            ``p58_serving_namespace``. The PVC is mounted at ``/mnt/persistent``.
        p02_verifier_model: Hugging Face model ID used by the model-download task only when
            ``p04_download_model`` is true. It defaults to empty. Configure ``hf-token`` if gated.
        p03_verifier_model_pvc: Verifier model directory relative to the PVC; set it to an
            existing model path or the directory populated when ``p04_download_model`` is true.
        p04_download_model: Download ``p02_verifier_model`` to the PVC before extraction. When false, the model
            must already exist at ``p03_verifier_model_pvc``.
        p05_dataset_name: Built-in dataset name or an existing JSON/JSONL dataset on the PVC. Supported built-ins
            are ``magpie``, ``ultrachat``, and ``gsm8k``; arbitrary Hugging Face IDs are not accepted here.
        p06_dataset_download_enabled: Forward ``p07_downloaded_dataset_uri`` to the Trainer SDK instead of
            ``p05_dataset_name``. Enable it for custom dataset URIs.
        p07_downloaded_dataset_uri: Dataset URI forwarded to Trainer when ``p06_dataset_download_enabled`` is true.
            It is required in that case; a PVC URI must reference ``p01_persistent_pvc``.
        p08_data_extraction_max_samples: Maximum extraction examples. Set to ``0`` to process the full dataset.
        p09_data_extraction_regenerate_responses: Regenerate assistant responses with the verifier instead of using
            original responses. Supported for the built-in datasets.
        p10_data_extraction_concurrency: Number of concurrent extraction requests. Use ``1`` on shared or NFS PVCs.
        p11_vllm_resource_gpu: GPUs requested by managed extraction vLLM. The SDK currently supports one GPU.
        p12_vllm_resource_memory: Memory requested by managed extraction vLLM.
        p13_vllm_gpu_memory_utilization: GPU memory fraction for managed extraction vLLM, in the range ``(0, 1]``.
        p14_vllm_readiness_timeout_minutes: Maximum wait for managed extraction vLLM readiness, in minutes.
        p15_extraction_runtime: ClusterTrainingRuntime used for the hidden-state extraction job.
        p16_output_dir: PVC-relative directory for extraction output, processed training data, and checkpoints. The
            final draft is copied to ``final_model`` at the PVC root.
        p17_hidden_states_path: PVC-relative hidden-state directory read by training. Managed extraction writes under
            ``p16_output_dir/hidden_states``; keep this path aligned with the output directory.
        p18_training_data_path: Directory containing the processed dataset written by extraction. Keep it aligned
            with ``p16_output_dir`` when changing the defaults.
        p19_total_seq_len: Positive maximum sequence length used for extraction, training, evaluation, and serving.
        p20_speculator_type: Draft architecture. The current Trainer and evaluator support ``eagle3``.
        p21_target_layer_ids: Four comma-separated verifier layer IDs. It defaults to empty; provide the IDs for the
            selected verifier.
        p22_training_epochs: Number of training epochs.
        p23_training_lr: Learning rate for draft-model training.
        p24_training_draft_vocab_size: Draft vocabulary size. ``0`` lets the Trainer SDK derive it.
        p25_training_num_layers: Number of layers in the draft model.
        p26_training_test_time_steps: Number of test-time training steps configured for Speculator.
        p27_training_norm_before_residual: Apply normalization before the residual connection.
        p28_training_norm_before_fc: Apply normalization before the fully connected layer.
        p29_training_embed_requires_grad: Whether draft embedding weights are trainable.
        p30_training_hidden_states_dtype: Hidden-state dtype; it must match the verifier model.
        p31_training_scheduler_type: Learning-rate scheduler: ``linear``, ``cosine``, or ``none``.
        p32_training_scheduler_warmup_steps: Warmup steps. ``0`` lets the SDK calculate them.
        p33_training_scheduler_total_steps: Total scheduler steps. ``0`` lets the SDK calculate them.
        p34_training_scheduler_num_cosine_cycles: Number of cosine cycles; ignored by other scheduler types.
        p35_training_checkpoint_freq: Checkpoint interval in epochs.
        p36_training_save_best: Save the best checkpoint by validation loss.
        p37_training_log_freq: Number of training steps between log updates.
        p38_training_resume_from_checkpoint: Resume from a saved checkpoint when supported by the runtime.
        p39_training_resource_cpu: CPU request for the training worker.
        p40_training_resource_gpu: Number of GPUs requested for the training worker.
        p41_training_resource_memory: Memory request for the training worker.
        p42_job_envs: Comma-separated ``KEY=VALUE`` environment overrides shared by extraction and training jobs.
        p43_job_runtime: ClusterTrainingRuntime used for draft training.
        p44_packages_to_install: Comma-separated Python packages installed in extraction and training jobs.
        p45_pip_index_urls: Comma-separated package-index URLs used when installing job packages.
        p46_evaluation_dataset_uri: Evaluation dataset ID or JSON/JSONL path; must be set;
            a ``pvc://`` URI must reference ``p01_persistent_pvc``.
        p47_evaluation_max_samples: Maximum GuideLLM requests per evaluation sweep point; must be positive.
        p48_evaluation_max_tokens: Maximum tokens generated per evaluation request; must be positive.
        p49_evaluation_temperature: Sampling temperature passed to GuideLLM. ``0`` is deterministic.
        p50_evaluation_num_speculative_tokens: Draft tokens proposed per step in evaluation and serving.
        p51_evaluation_gpu_memory_utilization: GPU memory fraction for evaluation and serving, greater than ``0`` and
            at most ``1.0``.
        p52_evaluation_startup_timeout_seconds: Maximum wait for the evaluation vLLM server to become ready.
        p53_evaluation_sample_seed: Seed used for deterministic sampling of local evaluation datasets.
        p54_model_registry_address: Model Registry hostname. Enter only the host, not the scheme or API path. Its
            default is empty, which skips registration.
        p55_model_registry_port: HTTP REST service port copied from the Model Registry URL; ``8080`` is common.
        p56_model_registry_name: Name assigned to the registered draft model; use a unique name in the registry.
        p57_model_registry_version: Version label for this registration. Use a new value for each revision.
        p58_serving_namespace: KServe namespace; must be set to access the PVC.
        p59_serving_endpoint_name: DNS-compatible name for the KServe InferenceService.
        p60_serving_runtime_image: Pinned compatible vLLM CUDA image pull spec used by KServe; override only with a
            compatible image.
    """
    persistent_pvc = p01_persistent_pvc
    verifier_model = p02_verifier_model
    verifier_model_pvc = p03_verifier_model_pvc
    download_model = p04_download_model
    dataset_name = p05_dataset_name
    dataset_download_enabled = p06_dataset_download_enabled
    downloaded_dataset_uri = p07_downloaded_dataset_uri
    data_extraction_max_samples = p08_data_extraction_max_samples
    data_extraction_regenerate_responses = p09_data_extraction_regenerate_responses
    data_extraction_concurrency = p10_data_extraction_concurrency
    vllm_resource_gpu = p11_vllm_resource_gpu
    vllm_resource_memory = p12_vllm_resource_memory
    vllm_gpu_memory_utilization = p13_vllm_gpu_memory_utilization
    vllm_readiness_timeout_minutes = p14_vllm_readiness_timeout_minutes
    extraction_runtime = p15_extraction_runtime
    output_dir = p16_output_dir
    hidden_states_path = p17_hidden_states_path
    training_data_path = p18_training_data_path
    total_seq_len = p19_total_seq_len
    speculator_type = p20_speculator_type
    target_layer_ids = p21_target_layer_ids
    training_epochs = p22_training_epochs
    training_lr = p23_training_lr
    training_draft_vocab_size = p24_training_draft_vocab_size
    training_num_layers = p25_training_num_layers
    training_ttt_steps = p26_training_test_time_steps
    training_norm_before_residual = p27_training_norm_before_residual
    training_norm_before_fc = p28_training_norm_before_fc
    training_embed_requires_grad = p29_training_embed_requires_grad
    training_hidden_states_dtype = p30_training_hidden_states_dtype
    training_scheduler_type = p31_training_scheduler_type
    training_scheduler_warmup_steps = p32_training_scheduler_warmup_steps
    training_scheduler_total_steps = p33_training_scheduler_total_steps
    training_scheduler_num_cosine_cycles = p34_training_scheduler_num_cosine_cycles
    training_checkpoint_freq = p35_training_checkpoint_freq
    training_save_best = p36_training_save_best
    training_log_freq = p37_training_log_freq
    training_resume_from_checkpoint = p38_training_resume_from_checkpoint
    training_resource_cpu = p39_training_resource_cpu
    training_resource_gpu = p40_training_resource_gpu
    training_resource_memory = p41_training_resource_memory
    job_envs = p42_job_envs
    training_runtime = p43_job_runtime
    packages_to_install = p44_packages_to_install
    pip_index_urls = p45_pip_index_urls
    evaluation_dataset_uri = p46_evaluation_dataset_uri
    evaluation_max_samples = p47_evaluation_max_samples
    evaluation_max_tokens = p48_evaluation_max_tokens
    evaluation_temperature = p49_evaluation_temperature
    num_speculative_tokens = p50_evaluation_num_speculative_tokens
    gpu_memory_utilization = p51_evaluation_gpu_memory_utilization
    evaluation_startup_timeout_seconds = p52_evaluation_startup_timeout_seconds
    evaluation_sample_seed = p53_evaluation_sample_seed
    model_registry_address = p54_model_registry_address
    model_registry_port = p55_model_registry_port
    model_registry_name = p56_model_registry_name
    model_registry_version = p57_model_registry_version
    serving_namespace = p58_serving_namespace
    serving_endpoint_name = p59_serving_endpoint_name
    serving_runtime_image = p60_serving_runtime_image

    model_download_task = download_model_component(
        model_name=verifier_model,
        model_cache_pvc=persistent_pvc,
        model_cache_mount="/mnt/persistent/models",
        download_enabled=download_model,
    )
    model_download_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(model_download_task, "IfNotPresent")
    kfp.kubernetes.mount_pvc(
        model_download_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    _attach_hf_token(model_download_task)

    extract_task = extract_speculator(
        verifier_model=verifier_model_pvc,
        verifier_model_pvc=verifier_model_pvc,
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset_name=dataset_name,
        dataset_download_enabled=dataset_download_enabled,
        downloaded_dataset_uri=downloaded_dataset_uri,
        hidden_states_path=hidden_states_path,
        total_seq_len=total_seq_len,
        speculator_type=speculator_type,
        target_layer_ids=target_layer_ids,
        data_extraction_max_samples=data_extraction_max_samples,
        data_extraction_regenerate_responses=data_extraction_regenerate_responses,
        data_extraction_concurrency=data_extraction_concurrency,
        vllm_resource_gpu=vllm_resource_gpu,
        vllm_resource_memory=vllm_resource_memory,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_readiness_timeout_minutes=vllm_readiness_timeout_minutes,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
        training_envs=job_envs,
        training_runtime=extraction_runtime,
        persistent_pvc=persistent_pvc,
        persistent_mount_path="/mnt/persistent",
    )
    extract_task.after(model_download_task)
    kfp.kubernetes.mount_pvc(
        extract_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    _attach_trainer_credentials(extract_task)

    train_task = train_speculator_mode(
        verifier_model=verifier_model_pvc,
        verifier_model_pvc=verifier_model_pvc,
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        hidden_states_path=hidden_states_path,
        training_data_path=training_data_path,
        total_seq_len=total_seq_len,
        speculator_type=speculator_type,
        target_layer_ids=target_layer_ids,
        training_epochs=training_epochs,
        training_lr=training_lr,
        training_draft_vocab_size=training_draft_vocab_size,
        training_num_layers=training_num_layers,
        training_ttt_steps=training_ttt_steps,
        training_norm_before_residual=training_norm_before_residual,
        training_norm_before_fc=training_norm_before_fc,
        training_embed_requires_grad=training_embed_requires_grad,
        training_hidden_states_dtype=training_hidden_states_dtype,
        training_scheduler_type=training_scheduler_type,
        training_scheduler_warmup_steps=training_scheduler_warmup_steps,
        training_scheduler_total_steps=training_scheduler_total_steps,
        training_scheduler_num_cosine_cycles=training_scheduler_num_cosine_cycles,
        training_checkpoint_freq=training_checkpoint_freq,
        training_save_best=training_save_best,
        training_log_freq=training_log_freq,
        training_resume_from_checkpoint=training_resume_from_checkpoint,
        training_resource_cpu=training_resource_cpu,
        training_resource_gpu=training_resource_gpu,
        training_resource_memory=training_resource_memory,
        data_extraction_concurrency=data_extraction_concurrency,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
        training_envs=job_envs,
        training_runtime=training_runtime,
        persistent_pvc=persistent_pvc,
        persistent_mount_path="/mnt/persistent",
    )
    train_task.after(extract_task)
    kfp.kubernetes.mount_pvc(
        train_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    _attach_trainer_credentials(train_task)

    evaluation_task = evaluate_speculator(
        verifier_model=verifier_model_pvc,
        draft_model_path="/mnt/persistent/final_model",
        evaluation_dataset_uri=evaluation_dataset_uri,
        evaluation_dataset_pvc=persistent_pvc,
        evaluation_speculator_type=speculator_type,
        evaluation_max_samples=evaluation_max_samples,
        evaluation_max_tokens=evaluation_max_tokens,
        evaluation_temperature=evaluation_temperature,
        evaluation_num_speculative_tokens=num_speculative_tokens,
        evaluation_gpu_memory_utilization=gpu_memory_utilization,
        evaluation_startup_timeout_seconds=evaluation_startup_timeout_seconds,
        evaluation_sample_seed=evaluation_sample_seed,
    )
    evaluation_task.after(train_task)
    kfp.kubernetes.mount_pvc(
        evaluation_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    evaluation_task.set_accelerator_type("nvidia.com/gpu")
    evaluation_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(evaluation_task, "IfNotPresent")
    _attach_hf_token(evaluation_task)

    model_registry_task = kubeflow_model_registry(
        pvc_mount_path="/mnt/persistent",
        model_pvc_name=persistent_pvc,
        input_model=train_task.outputs["output_model"],
        input_metrics=train_task.outputs["output_metrics"],
        eval_metrics=evaluation_task.outputs["output_metrics"],
        eval_results=evaluation_task.outputs["output_results"],
        registry_address=model_registry_address,
        registry_port=model_registry_port,
        model_name=model_registry_name,
        model_version=model_registry_version,
        model_description="Speculator draft model trained by the offline pipeline.",
        shared_log_file="pipeline_log.txt",
        source_pipeline_name=PIPELINE_NAME,
        source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        source_namespace="",
    )
    model_registry_task.after(evaluation_task)
    kfp.kubernetes.mount_pvc(
        model_registry_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    model_registry_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(model_registry_task, "IfNotPresent")

    serving_task = speculative_decoding_serving(
        namespace=serving_namespace,
        endpoint_name=serving_endpoint_name,
        model_cache_pvc=persistent_pvc,
        verifier_model_dir=verifier_model_pvc,
        draft_model_dir="final_model",
        runtime_image=serving_runtime_image,
        max_model_len=total_seq_len,
        num_speculative_tokens=num_speculative_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    serving_task.after(model_registry_task)
    serving_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(serving_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=speculator_offline_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )

"""Train, evaluate, register, and serve a Speculator draft model."""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.deployment.speculative_decoding_serving import speculative_decoding_serving
from components.evaluation.speculator import evaluate_speculator
from components.training.finetuning.speculator import train_speculator_mode

PVC_SIZE = "200Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "speculator-train-only-pipeline"


def _attach_k8s(task) -> None:
    """Attach the Kubernetes credentials and Hugging Face token to a task."""
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
    kfp.kubernetes.use_secret_as_env(
        task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Train and evaluate an Eagle3 draft from PVC-stored artifacts, register it, and serve it.",
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
def speculator_train_only_pipeline(
    persistent_pvc: str,
    serving_namespace: str,
    model_registry_address: str,
    model_registry_name: str,
    serving_endpoint_name: str,
    model_registry_port: int = 8080,
    model_registry_version: str = "1.0.0",
    serving_runtime_image: str = "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a",  # noqa: E501
    evaluation_dataset_uri: str = "RedHatAI/speculator_benchmarks",
    verifier_model_pvc: str = "models/Qwen--Qwen3-0.6B",
    output_dir: str = "speculator-output",
    hidden_states_path: str = "speculator-output/hidden_states",
    training_data_path: str = "speculator-output",
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: str = "2,14,25,28",
    training_epochs: int = 3,
    training_lr: float = 1e-4,
    training_draft_vocab_size: int = 0,
    training_num_layers: int = 1,
    training_ttt_steps: int = 1,
    training_norm_before_residual: bool = True,
    training_norm_before_fc: bool = False,
    training_embed_requires_grad: bool = False,
    training_hidden_states_dtype: str = "bfloat16",
    training_scheduler_type: str = "linear",
    training_scheduler_warmup_steps: int = 0,
    training_scheduler_total_steps: int = 0,
    training_scheduler_num_cosine_cycles: float = 0.5,
    training_checkpoint_freq: float = 1.0,
    training_save_best: bool = False,
    training_log_freq: int = 1,
    training_resume_from_checkpoint: bool = False,
    training_resource_cpu: str = "4",
    training_resource_gpu: int = 1,
    training_resource_memory: str = "64Gi",
    enable_progression_tracking: bool = True,
    metrics_port: int = 28080,
    metrics_poll_interval_seconds: int = 30,
    packages_to_install: str = "",
    pip_index_urls: str = "",
    training_envs: str = "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1",
    training_runtime: str = "speculator-model-opt-cuda",
    evaluation_max_samples: int = 80,
    evaluation_max_tokens: int = 256,
    evaluation_temperature: float = 0.0,
    evaluation_num_speculative_tokens: int = 5,
    evaluation_gpu_memory_utilization: float = 0.9,
    evaluation_startup_timeout_seconds: int = 900,
    evaluation_sample_seed: int = 42,
):
    """Train, evaluate, register, and serve a draft from existing artifacts.

    This pipeline runs Trainer's train-only mode and expects the verifier model,
    hidden states, and preprocessed Arrow data to already exist on the shared PVC.
    It does not download the verifier. Relative storage paths are resolved against
    ``persistent_pvc``. Explicit ``pvc://<claim>/<path>`` paths must reference that
    same PVC.

    For example, with ``persistent_pvc="shared"``,
    ``verifier_model_pvc="models/Qwen--Qwen3-0.6B"`` is equivalent to
    ``pvc://shared/models/Qwen--Qwen3-0.6B``. Likewise,
    ``output_dir="speculator-output"`` and
    ``hidden_states_path="speculator-output/hidden_states"`` can be written as
    ``pvc://shared/speculator-output`` and
    ``pvc://shared/speculator-output/hidden_states``. The default
    ``training_data_path`` and ``output_dir`` are both ``speculator-output``,
    matching the data-only pipeline defaults. If you customize the data-only
    output directory, update ``training_data_path`` and ``hidden_states_path`` too.
    The hidden-state default matches the data-only pipeline's default output
    directory; it is not discovered automatically from a previous pipeline run.

    The trained draft is copied to ``pvc://<persistent_pvc>/final_model`` for
    evaluation, Model Registry, and serving. The resulting KServe endpoint uses
    cluster-local visibility.

    Model Registry setup:
        In the RHOAI dashboard, open **AI Hub > Models > Model registries** and
        create or select a registry. Copy its HTTP REST service URL. This pipeline
        takes the host and port separately: put only the hostname in
        ``model_registry_address`` and the service port in ``model_registry_port``.
        For a URL such as
        ``http://registry-service.registry-namespace.svc:8080/api/model_registry/v1alpha3``,
        use ``registry-service.registry-namespace.svc`` and ``8080``; omit the
        scheme and API path. The pipeline ServiceAccount must have a RoleBinding
        to ``registry-user-<registry-name>`` in the registry namespace.
        ``model_registry_address`` is required because this pipeline registers the
        evaluated model before serving it. Use a new ``model_registry_version`` for
        each trained revision.

    Args:
        persistent_pvc: Name only of an existing ReadWriteMany PVC mounted at
            ``/mnt/persistent``; do not include a path or ``pvc://`` prefix. It
            stores the verifier, hidden states, preprocessed data, and model output.
        serving_namespace: OpenShift namespace where KServe creates the endpoint.
            It must be the PVC's namespace and have the required KServe permissions.
        model_registry_address: Required HTTP REST host from AI Hub > Models > Model registries.
            Enter only the hostname, without scheme or API path.
        model_registry_port: HTTP REST service port copied from the registry endpoint; ``8080`` is the default.
        model_registry_name: Required model name; choose a name unique within the selected registry.
        model_registry_version: Version label for this registration. Use a new version for each trained revision.
        serving_runtime_image: Eagle3-compatible vLLM image pull spec. The default is
            the pinned Red Hat runtime; override only with a verified digest.
        evaluation_dataset_uri: Hugging Face dataset ID. The supported default,
            ``RedHatAI/speculator_benchmarks``, runs its ``HumanEval`` subset. For
            local data, provide an existing JSONL file on ``persistent_pvc`` as a
            relative path or matching ``pvc://`` URI.
        serving_endpoint_name: Required DNS-compatible KServe InferenceService name, unique in ``serving_namespace``.
        verifier_model_pvc: Required existing verifier directory on
            ``persistent_pvc``; the default is ``models/Qwen--Qwen3-0.6B``. Use a
            relative path or ``pvc://<persistent_pvc>/<path>``. Training, evaluation,
            and serving all read this directory.
        output_dir: Trainer checkpoint/output directory on ``persistent_pvc``.
            Default: ``speculator-output``. Use a relative path or matching URI.
        hidden_states_path: Existing hidden-state directory from data-only extraction.
            The default ``speculator-output/hidden_states`` matches the data-only
            default ``output_dir="speculator-output"``; it is not auto-detected.
            Override it if the data-only run used a different output directory.
            Use a relative path or matching ``pvc://`` URI.
        training_data_path: Preprocessed Arrow dataset directory from data-only
            extraction. Default: ``speculator-output``. Set it to the data-only
            output path (relative or matching ``pvc://`` URI); an empty value falls
            back to ``output_dir``.
        total_seq_len: Positive sequence length for preprocessing and training; keep
            it consistent with the data-only extraction run.
        speculator_type: Draft architecture. ``eagle3`` is currently the supported training type.
        target_layer_ids: Exactly four comma-separated verifier layer indices. Default
            ``2,14,25,28`` matches the default 28-layer Qwen3-0.6B model. Use the same
            four indices as data-only extraction; set matching indices for another verifier.
        training_epochs: Number of training epochs; must be positive.
        training_lr: Positive learning rate for draft-model training.
        training_draft_vocab_size: Draft vocabulary size. ``0`` lets the SDK derive it from the verifier vocabulary.
        training_num_layers: Number of layers in the draft model.
        training_ttt_steps: Number of test-time-training steps.
        training_norm_before_residual: Apply normalization before the residual connection.
        training_norm_before_fc: Apply normalization before the fully connected layer.
        training_embed_requires_grad: Whether the draft embedding weights are trainable.
        training_hidden_states_dtype: Hidden-state tensor dtype: ``bfloat16``,
            ``float16``, or ``float32``; it must match the verifier.
        training_scheduler_type: Learning-rate scheduler: ``linear``, ``cosine``, or ``none``.
        training_scheduler_warmup_steps: Warmup steps; ``0`` lets the SDK compute 1% of total training steps.
        training_scheduler_total_steps: Total scheduler steps; ``0`` lets the SDK
            calculate the value from epochs and dataset size.
        training_scheduler_num_cosine_cycles: Cosine cycles; used only when ``training_scheduler_type="cosine"``.
        training_checkpoint_freq: Checkpoint interval in epochs.
        training_save_best: Save only the best checkpoint by validation loss.
        training_log_freq: Training steps between log updates.
        training_resume_from_checkpoint: Resume training from a checkpoint when the runtime supports it.
        training_resource_cpu: CPU request for the TrainJob worker, as a Kubernetes quantity.
        training_resource_gpu: Number of GPUs requested for the TrainJob worker.
        training_resource_memory: Memory request for the TrainJob worker, as a Kubernetes quantity.
        enable_progression_tracking: Enable Trainer progress reporting.
        metrics_port: HTTP port for training metrics; valid range is ``1024``-``65535``.
        metrics_poll_interval_seconds: Seconds between metrics polls; valid range is ``5``-``300``.
        packages_to_install: Comma-separated Python packages to install in the TrainJob.
        pip_index_urls: Comma-separated package-index URLs for TrainJob installs.
        training_envs: Comma-separated ``KEY=VALUE`` environment overrides for the TrainJob.
        training_runtime: Name of the ClusterTrainingRuntime used by the TrainJob.
        evaluation_max_samples: Maximum requests per evaluation sweep point; must be positive.
        evaluation_max_tokens: Maximum generated tokens per evaluation request; must be positive.
        evaluation_temperature: Sampling temperature used by GuideLLM; ``0`` is deterministic.
        evaluation_num_speculative_tokens: Number of draft tokens proposed per step; also used by serving.
        evaluation_gpu_memory_utilization: GPU memory fraction for evaluation and serving; range ``(0, 1]``.
        evaluation_startup_timeout_seconds: Maximum seconds to wait for the evaluation vLLM server to become ready.
        evaluation_sample_seed: Seed for deterministic sampling from a local evaluation dataset.
    """
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
        enable_progression_tracking=enable_progression_tracking,
        metrics_port=metrics_port,
        metrics_poll_interval_seconds=metrics_poll_interval_seconds,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
        training_envs=training_envs,
        training_runtime=training_runtime,
        persistent_pvc=persistent_pvc,
        persistent_mount_path="/mnt/persistent",
    )
    kfp.kubernetes.mount_pvc(train_task, pvc_name=persistent_pvc, mount_path="/mnt/persistent")
    _attach_k8s(train_task)

    evaluation_task = evaluate_speculator(
        verifier_model=verifier_model_pvc,
        draft_model_path="/mnt/persistent/final_model",
        evaluation_dataset_uri=evaluation_dataset_uri,
        evaluation_dataset_pvc=persistent_pvc,
        evaluation_max_samples=evaluation_max_samples,
        evaluation_max_tokens=evaluation_max_tokens,
        evaluation_temperature=evaluation_temperature,
        evaluation_num_speculative_tokens=evaluation_num_speculative_tokens,
        evaluation_speculator_type=speculator_type,
        evaluation_gpu_memory_utilization=evaluation_gpu_memory_utilization,
        evaluation_startup_timeout_seconds=evaluation_startup_timeout_seconds,
        evaluation_sample_seed=evaluation_sample_seed,
    )
    evaluation_task.after(train_task)
    kfp.kubernetes.mount_pvc(evaluation_task, pvc_name=persistent_pvc, mount_path="/mnt/persistent")
    evaluation_task.set_accelerator_type("nvidia.com/gpu")
    _attach_k8s(evaluation_task)

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
        model_description="Speculator draft model trained by the train-only pipeline.",
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
        num_speculative_tokens=evaluation_num_speculative_tokens,
        gpu_memory_utilization=evaluation_gpu_memory_utilization,
    )
    serving_task.after(model_registry_task)
    serving_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(serving_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=speculator_train_only_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )

"""Managed-vLLM Speculator hidden-state extraction pipeline."""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.download_model import download_model as download_model_component
from components.training.finetuning.speculator import extract_speculator

PVC_SIZE = "200Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "speculator-data-only-pipeline"


def _attach_k8s(task) -> None:
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
    description="Extract Speculator hidden states using a managed vLLM sidecar.",
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
def speculator_data_only_pipeline(
    persistent_pvc: str,
    download_model: bool = False,
    vllm_source: str = "managed",
    verifier_model: str = "",
    verifier_model_pvc: str = "models/Qwen--Qwen2.5-1.5B-Instruct",
    output_dir: str = "speculator-output",
    dataset_name: str = "ultrachat",
    hidden_states_path: str = "speculator-output/hidden_states",
    vllm_endpoint: str = "",
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: str = "",
    data_extraction_max_samples: int = 16,
    data_extraction_regenerate_responses: bool = False,
    data_extraction_concurrency: int = 1,
    vllm_resource_gpu: int = 1,
    vllm_resource_memory: str = "96Gi",
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_readiness_timeout_minutes: int = 60,
    enable_progression_tracking: bool = True,
    metrics_port: int = 28080,
    metrics_poll_interval_seconds: int = 30,
    packages_to_install: str = "",
    pip_index_urls: str = "",
    training_envs: str = "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1",
    training_runtime: str = "vllm-extract-cuda",
):
    """Extract verifier hidden states using managed or external vLLM.

    Managed mode starts a vLLM sidecar. When ``download_model`` is false, it uses
    ``verifier_model_pvc`` as the cached verifier path; when true, it downloads
    ``verifier_model`` to the PVC. Remote mode calls an existing vLLM endpoint and
    uses ``verifier_model_pvc`` as the model path visible to that service.
    Hidden states are written below ``output_dir`` on the configured storage.

    Args:
        persistent_pvc: RWX PVC for models, data, and hidden states.
        download_model: Download ``verifier_model`` to the PVC when True; otherwise
            use the cached ``verifier_model_pvc`` path in managed mode.
        vllm_source: Choose ``managed`` sidecar or ``remote`` external vLLM.
        verifier_model: Hugging Face verifier ID to download; leave empty to reuse the PVC model.
        verifier_model_pvc: Cached PVC model path for managed mode, or model path for remote mode.
        output_dir: PVC-relative output directory.
        dataset_name: Built-in ``magpie``, ``ultrachat``, ``gsm8k``, or a PVC JSON/JSONL URI.
        hidden_states_path: PVC hidden-state path for remote mode.
        vllm_endpoint: External vLLM URL; required when source is ``remote``.
        total_seq_len: Maximum preprocessing and extraction sequence length.
        speculator_type: Draft type; currently ``eagle3``.
        target_layer_ids: Four layer IDs for a PVC verifier; empty enables SDK auto-selection for HF models.
        data_extraction_max_samples: Samples to process; zero means all.
        data_extraction_regenerate_responses: Regenerate built-in dataset responses with the verifier.
        data_extraction_concurrency: Concurrent requests; use ``1`` on shared/NFS PVCs.
        vllm_resource_gpu: GPUs for managed vLLM.
        vllm_resource_memory: Memory for managed vLLM.
        vllm_gpu_memory_utilization: vLLM GPU memory fraction.
        vllm_readiness_timeout_minutes: vLLM startup timeout.
        enable_progression_tracking: Enable progress reporting.
        metrics_port: Progress metrics port.
        metrics_poll_interval_seconds: Progress polling interval.
        packages_to_install: Comma-separated TrainJob packages.
        pip_index_urls: Comma-separated TrainJob package indexes.
        training_envs: Comma-separated TrainJob environment overrides.
        training_runtime: ClusterTrainingRuntime for extraction.
    """
    common_kwargs = dict(
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset_name=dataset_name,
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
        enable_progression_tracking=enable_progression_tracking,
        metrics_port=metrics_port,
        metrics_poll_interval_seconds=metrics_poll_interval_seconds,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
        training_envs=training_envs,
        training_runtime=training_runtime,
        persistent_pvc=persistent_pvc,
        persistent_mount_path="/mnt/persistent",
        download_model=download_model,
        verifier_model_pvc=verifier_model_pvc,
    )

    download_task = download_model_component(
        model_name=verifier_model,
        model_cache_pvc=persistent_pvc,
        model_cache_mount="/mnt/persistent/models",
        download_enabled=download_model,
    )
    download_task.set_caching_options(False)
    kfp.kubernetes.use_secret_as_env(
        download_task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )
    kfp.kubernetes.mount_pvc(
        download_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )

    with dsl.If(vllm_source == "managed", name="vllm-connected"):
        managed_task = extract_speculator(
            verifier_model=verifier_model,
            vllm_source="managed",
            **common_kwargs,
        )
        managed_task.after(download_task)
        kfp.kubernetes.mount_pvc(
            managed_task,
            pvc_name=persistent_pvc,
            mount_path="/mnt/persistent",
        )
        _attach_k8s(managed_task)

    with dsl.Else(name="vllm-disconnected"):
        remote_task = extract_speculator(
            verifier_model=verifier_model_pvc,
            vllm_source="remote",
            vllm_endpoint=vllm_endpoint,
            **common_kwargs,
        )
        remote_task.after(download_task)
        kfp.kubernetes.mount_pvc(
            remote_task,
            pvc_name=persistent_pvc,
            mount_path="/mnt/persistent",
        )
        _attach_k8s(remote_task)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=speculator_data_only_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )

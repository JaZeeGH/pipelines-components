"""Speculator hidden-state extraction pipeline with managed or remote vLLM."""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.download_model import download_model as download_model_component
from components.training.finetuning.speculator import extract_hidden_states

PVC_SIZE = "200Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "speculator-hidden-state-extraction-pipeline"


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
    description="Extract verifier hidden states for speculative decoding training using managed or remote vLLM.",
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
def speculator_hidden_state_extraction_pipeline(
    persistent_pvc: str,
    download_model: bool = False,
    vllm_mode: str = "managed",
    verifier_model: str = "",
    verifier_model_pvc: str = "",
    output_dir: str = "speculator-output",
    dataset_name: str = "ultrachat",
    hidden_states_path: str = "speculator-output/hidden_states",
    vllm_endpoint: str = "",
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: str = "",
    max_samples: int = 16,
    regenerate_responses: bool = False,
    data_extraction_concurrency: int = 1,
    vllm_resource_gpu: int = 1,
    vllm_resource_memory: str = "96Gi",
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_readiness_timeout_minutes: int = 60,
    packages_to_install: str = "",
    pip_index_urls: str = "",
    job_envs: str = "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1",
    job_runtime: str = "vllm-extract-cuda",
):
    """Extract verifier hidden states for speculative decoding training using managed or remote vLLM.

    Managed mode runs vLLM in the Trainer job. When ``download_model`` is true,
    ``verifier_model`` is downloaded to ``persistent_pvc``; otherwise, managed mode
    reuses ``verifier_model_pvc``. Remote mode uses ``vllm_endpoint`` and requires
    the verifier and hidden-state PVC paths to be accessible to the external service.
    Managed mode writes hidden states under ``output_dir/hidden_states``; remote
    mode writes them to ``hidden_states_path``.

    Relative storage paths are resolved against ``persistent_pvc``. For example,
    with ``persistent_pvc="shared"``, ``output_dir="speculator-output"`` resolves
    to ``pvc://shared/speculator-output``. ``speculator-output`` is the default
    directory name and can be changed.

    Args:
        persistent_pvc: Existing PVC mounted at ``/mnt/persistent`` for the verifier,
            local datasets, and outputs.
        download_model: Download ``verifier_model`` to the PVC. Remote mode uses
            ``verifier_model_pvc``, which must point to the downloaded or cached model.
        vllm_mode: Use ``managed`` for a vLLM sidecar in the Trainer job or ``remote`` for an external vLLM service.
        verifier_model: Hugging Face Hub model ID to download when ``download_model`` is
            true; leave empty when reusing ``verifier_model_pvc``.
        verifier_model_pvc: Path to the cached or downloaded verifier on the PVC;
            required when reusing a cached model or using remote vLLM. Use a relative
            path or a ``pvc://`` URI.
        output_dir: Trainer output directory on the PVC. ``speculator-output`` is the
            default; choose another relative path or a URI for that PVC.
        dataset_name: Use ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered),
            ``ultrachat`` (HuggingFaceH4/ultrachat_200k; default), or ``gsm8k``
            (openai/gsm8k). For custom data, use an existing JSON/JSONL file on the
            PVC (relative path or ``pvc://<claim>/<path>``); arbitrary Hugging Face
            dataset IDs are not supported.
        hidden_states_path: Remote-mode hidden-state output on the PVC shared with
            vLLM; managed mode writes to ``<output_dir>/hidden_states``.
        vllm_endpoint: URL of the external vLLM service; required for ``remote`` and blank for ``managed``.
        total_seq_len: Maximum sequence length for dataset preprocessing and vLLM extraction; must be positive.
        speculator_type: Draft architecture; ``eagle3`` is currently the only supported type.
        target_layer_ids: Exactly four comma-separated verifier layer indices; required
            for the default PVC-backed verifier and remote vLLM. The SDK can infer them
            only from a Hugging Face model ID passed directly.
        max_samples: Maximum samples to process; ``0`` processes the full dataset.
        regenerate_responses: Regenerate responses with the verifier;
            supported only for ``magpie``, ``ultrachat``, and ``gsm8k``.
        data_extraction_concurrency: Number of concurrent vLLM requests for
            hidden-state extraction; use ``1`` on shared or NFS storage.
        vllm_resource_gpu: GPUs requested by managed vLLM; the SDK currently supports exactly ``1``.
        vllm_resource_memory: Memory requested by the managed vLLM sidecar.
        vllm_gpu_memory_utilization: Fraction of GPU memory available to vLLM, in the range ``(0, 1]``.
        vllm_readiness_timeout_minutes: Maximum wait for the managed or remote vLLM
            endpoint to become ready, in minutes; minimum ``1``.
        packages_to_install: Comma-separated Python packages to install in the Trainer job.
        pip_index_urls: Comma-separated package-index URLs used for Trainer job installs.
        job_envs: Comma-separated ``KEY=VALUE`` environment overrides for the extraction job.
        job_runtime: Name of the cluster runtime used by the extraction job.
    """
    common_kwargs = dict(
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset_name=dataset_name,
        hidden_states_path=hidden_states_path,
        total_seq_len=total_seq_len,
        speculator_type=speculator_type,
        target_layer_ids=target_layer_ids,
        data_extraction_max_samples=max_samples,
        data_extraction_regenerate_responses=regenerate_responses,
        data_extraction_concurrency=data_extraction_concurrency,
        vllm_resource_gpu=vllm_resource_gpu,
        vllm_resource_memory=vllm_resource_memory,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_readiness_timeout_minutes=vllm_readiness_timeout_minutes,
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
        training_envs=job_envs,
        training_runtime=job_runtime,
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

    with dsl.If(vllm_mode == "managed", name="managed-vllm"):
        managed_task = extract_hidden_states(
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

    with dsl.Else(name="remote-vllm"):
        remote_task = extract_hidden_states(
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
        pipeline_func=speculator_hidden_state_extraction_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )

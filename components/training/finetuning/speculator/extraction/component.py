"""Mode-specific Speculator hidden-state extraction component."""

import os
from typing import Optional

from kfp import dsl

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")


def _resolve_verifier_model_pvc_uri(verifier_model_pvc: str, persistent_pvc: str) -> str:
    """Normalize a verifier directory to a URI for an external vLLM Trainer job."""
    verifier_model_pvc = verifier_model_pvc.strip()
    if not verifier_model_pvc:
        raise ValueError("verifier_model_pvc is required when vllm_source='remote'")
    if not persistent_pvc:
        raise ValueError("persistent_pvc is required when vllm_source='remote'")

    if verifier_model_pvc.startswith("pvc://"):
        claim, separator, path = verifier_model_pvc.removeprefix("pvc://").partition("/")
        if not separator or not path:
            raise ValueError("Verifier PVC URIs must use pvc://<claim>/<path>")
        if claim != persistent_pvc:
            raise ValueError(f"Verifier PVC URI must reference persistent_pvc {persistent_pvc!r}")
    else:
        path = verifier_model_pvc

    path_parts = path.strip("/").split("/")
    if path.startswith("/") or any(part in {"", ".", ".."} for part in path_parts):
        raise ValueError("verifier_model_pvc must be a relative PVC path without traversal")

    relative_path = "/".join(path_parts)
    return f"pvc://{persistent_pvc}/{relative_path}"


@dsl.component(
    base_image=(
        "quay.io/opendatahub/odh-th-torch-cuda-py312:odh-3.6-ea.1@"
        "sha256:9ad2d72ebe892dffd3554eeb81e2a47c9fcf0d97f89ed261d48f4e8bbc367b4b"
    ),
    packages_to_install=["kfp==2.17.0", "kubernetes", "olot"],
    install_kfp_package=False,
    embedded_artifact_path=_SHARED_DIR,
    additional_funcs=[_resolve_verifier_model_pvc_uri],
    task_config_passthroughs=[
        dsl.TaskConfigField.RESOURCES,
        dsl.TaskConfigField.KUBERNETES_TOLERATIONS,
        dsl.TaskConfigField.KUBERNETES_NODE_SELECTOR,
        dsl.TaskConfigField.KUBERNETES_AFFINITY,
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.ENV, apply_to_task=True),
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.KUBERNETES_VOLUMES, apply_to_task=True),
    ],
)
def extract_speculator(
    verifier_model: str,
    output_dir: str,
    pvc_path: str,
    dataset_name: str,
    verifier_model_pvc: Optional[str] = None,
    vllm_source: str = "managed",
    dataset_download_enabled: bool = False,
    downloaded_dataset_uri: Optional[str] = None,
    hidden_states_path: Optional[str] = None,
    vllm_endpoint: Optional[str] = None,
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: Optional[str] = None,
    data_extraction_max_samples: Optional[int] = None,
    data_extraction_regenerate_responses: bool = False,
    data_extraction_concurrency: int = 4,
    vllm_resource_gpu: int = 1,
    vllm_resource_memory: str = "96Gi",
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_readiness_timeout_minutes: int = 60,
    enable_progression_tracking: bool = True,
    metrics_port: int = 28080,
    metrics_poll_interval_seconds: int = 30,
    packages_to_install: Optional[str] = None,
    pip_index_urls: Optional[str] = None,
    training_envs: str = "",
    training_runtime: str = "speculator-model-opt-cuda",
    persistent_pvc: str = "",
    persistent_mount_path: str = "/mnt/persistent",
    download_model: bool = False,
    output_hidden_states: dsl.Output[dsl.Artifact] = None,
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Extract verifier hidden states using managed or external vLLM.

    ``dataset_name`` selects the source dataset. When regeneration is enabled,
    responses are regenerated through the verifier before hidden-state extraction.
    Managed mode uses the in-job vLLM sidecar; providing ``vllm_endpoint`` uses an
    external vLLM service and requires matching shared model and hidden-state paths.
    """
    from speculator import run_speculator

    vllm_source = vllm_source.strip().lower()
    if vllm_source not in {"managed", "remote"}:
        raise ValueError("vllm_source must be either 'managed' or 'remote'")
    vllm_endpoint = (vllm_endpoint or "").strip()
    if vllm_source == "remote":
        if not vllm_endpoint:
            raise ValueError("vllm_endpoint is required when vllm_source='remote'")
        verifier_model_pvc = _resolve_verifier_model_pvc_uri(verifier_model_pvc or verifier_model, persistent_pvc)
        verifier_model = verifier_model_pvc
    elif vllm_endpoint:
        raise ValueError("vllm_endpoint must be blank when vllm_source='managed'")
    if not vllm_endpoint:
        vllm_endpoint = None
    if dataset_download_enabled:
        if not downloaded_dataset_uri:
            raise ValueError("downloaded_dataset_uri is required when dataset download is enabled")
        dataset_name = downloaded_dataset_uri

    return run_speculator("data_only", locals())

"""Mode-specific Speculator hidden-state extraction component."""

import os
from typing import Optional

from kfp import dsl

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")


@dsl.component(
    base_image=(
        "quay.io/opendatahub/odh-th-torch-cuda-py312:odh-3.6-ea.1@"
        "sha256:9ad2d72ebe892dffd3554eeb81e2a47c9fcf0d97f89ed261d48f4e8bbc367b4b"
    ),
    packages_to_install=["kfp==2.17.0", "kubernetes", "olot"],
    install_kfp_package=False,
    embedded_artifact_path=_SHARED_DIR,
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
    Managed mode uses the in-job vLLM sidecar. When model downloading is disabled,
    the managed sidecar uses ``verifier_model_pvc`` if provided. Remote mode uses
    an external vLLM service and requires matching shared model and hidden-state paths.
    Relative and absolute JSON/JSONL paths under ``persistent_mount_path`` are
    resolved to a PVC URI before the Trainer job starts. Existing
    ``pvc://<claim>/<path>`` dataset references retain their Trainer URI semantics.

    ``verifier_model_pvc``, ``output_dir``, and ``hidden_states_path`` accept
    paths relative to the configured PVC mount or explicit
    ``pvc://<claim>/<path>`` Trainer URIs. For example, with
    ``persistent_pvc="shared"``, ``verifier_model_pvc="models/verifier"`` and
    ``verifier_model_pvc="pvc://shared/models/verifier"`` identify the same model;
    ``output_dir="speculator-output"`` and ``output_dir="pvc://shared/speculator-output"``
    identify the same directory; ``hidden_states_path="speculator-output/hidden_states"``
    and ``hidden_states_path="pvc://shared/speculator-output/hidden_states"`` identify
    the same hidden-state path. Relative paths use ``persistent_pvc`` when supplied,
    otherwise the KFP workspace PVC.
    """
    from pathlib import Path

    def resolve_local_dataset_path(dataset_uri: str) -> str:
        if not dataset_uri:
            return dataset_uri

        if dataset_uri.startswith("pvc://"):
            claim, separator, dataset_path = dataset_uri.removeprefix("pvc://").partition("/")
            if not claim or not separator or not dataset_path:
                raise ValueError("PVC dataset URIs must use pvc://<claim>/<path>")
            requested_path = Path(dataset_path)
            if claim != persistent_pvc:
                if requested_path.is_absolute() or ".." in requested_path.parts:
                    raise ValueError("PVC dataset paths must remain relative")
                if requested_path.suffix.lower() not in {".json", ".jsonl"}:
                    raise ValueError("dataset files on the shared PVC must end in .json or .jsonl")
                return dataset_uri
        elif "://" in dataset_uri:
            return dataset_uri

        else:
            candidate = Path(dataset_uri)
            if candidate.suffix.lower() not in {".json", ".jsonl"}:
                return dataset_uri
            dataset_path = dataset_uri

        if not persistent_pvc:
            raise ValueError("persistent_pvc is required for local JSON/JSONL dataset paths")

        pvc_root = Path(persistent_mount_path).resolve()
        candidate = Path(dataset_path)
        candidate = candidate if candidate.is_absolute() else pvc_root / candidate
        resolved = candidate.resolve()
        try:
            relative_path = resolved.relative_to(pvc_root)
        except ValueError as exc:
            raise ValueError("local dataset paths must remain within persistent_mount_path") from exc

        if not relative_path.parts:
            raise ValueError("dataset path must be a JSON or JSONL file under persistent_mount_path")
        if relative_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("dataset files on the shared PVC must end in .json or .jsonl")
        if not resolved.is_file():
            raise FileNotFoundError(f"dataset file not found on persistent_pvc: {resolved}")

        return f"pvc://{persistent_pvc}/{relative_path.as_posix()}"

    if vllm_source == "remote":
        verifier_model = verifier_model_pvc or verifier_model
    elif verifier_model_pvc and not download_model:
        verifier_model = verifier_model_pvc
    if not vllm_endpoint:
        vllm_endpoint = None
    if dataset_download_enabled:
        if not downloaded_dataset_uri:
            raise ValueError("downloaded_dataset_uri is required when dataset download is enabled")
        dataset_name = downloaded_dataset_uri
    else:
        dataset_name = resolve_local_dataset_path(dataset_name)

    from speculator import run_speculator

    return run_speculator("data_only", locals())

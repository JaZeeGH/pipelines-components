"""Speculator offline extraction and training pipeline."""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.deployment.speculative_decoding_serving import speculative_decoding_serving
from components.evaluation.speculator import evaluate_speculator
from components.training.finetuning.speculator import extract_speculator, train_speculator_mode

PVC_SIZE = "200Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "speculator-offline-pipeline"


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
    description=(
        "Extract hidden states with external vLLM, then train, evaluate, register, and serve a Speculator model."
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
    persistent_pvc: str,
    serving_namespace: str,
    serving_runtime_image: str,
    evaluation_dataset_uri: str = "RedHatAI/speculator_benchmarks",
    model_registry_address: str = "",
    model_registry_port: int = 8080,
    model_registry_name: str = "speculator-offline-draft",
    model_registry_version: str = "1.0.0",
    serving_endpoint_name: str = "speculator-offline",
    verifier_model: str = "models/Qwen--Qwen2.5-1.5B-Instruct",
    output_dir: str = "speculator-output",
    dataset_name: str = "ultrachat",
    hidden_states_path: str = "speculator-output/hidden_states",
    vllm_endpoint: str = "",
    total_seq_len: int = 2048,
    speculator_type: str = "eagle3",
    target_layer_ids: str = "",
    data_extraction_max_samples: int = 16,
    data_extraction_regenerate_responses: bool = False,
    data_extraction_concurrency: int = 4,
    training_epochs: int = 1,
    training_lr: float = 0.0001,
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
    """Extract hidden states, train and evaluate the draft, register it, then serve it.

    Set `model_registry_address` and `model_registry_port` to connect to Model Registry.
    Registration is skipped when the address is empty; increment `model_registry_version` for each new version.
    Serving requires the namespace and a verified vLLM runtime image pull spec.
    """
    extract_task = extract_speculator(
        verifier_model=verifier_model,
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset_name=dataset_name,
        hidden_states_path=hidden_states_path,
        vllm_endpoint=vllm_endpoint,
        total_seq_len=total_seq_len,
        speculator_type=speculator_type,
        target_layer_ids=target_layer_ids,
        data_extraction_max_samples=data_extraction_max_samples,
        data_extraction_regenerate_responses=data_extraction_regenerate_responses,
        data_extraction_concurrency=data_extraction_concurrency,
        training_envs=training_envs,
        training_runtime=training_runtime,
        persistent_pvc=persistent_pvc,
        persistent_mount_path="/mnt/persistent",
        packages_to_install=packages_to_install,
        pip_index_urls=pip_index_urls,
    )
    kfp.kubernetes.mount_pvc(
        extract_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    _attach_k8s(extract_task)

    task = train_speculator_mode(
        verifier_model=verifier_model,
        verifier_model_pvc=verifier_model,
        output_dir=output_dir,
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        hidden_states_path=hidden_states_path,
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
    kfp.kubernetes.mount_pvc(
        task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    task.after(extract_task)
    _attach_k8s(task)

    evaluation_task = evaluate_speculator(
        verifier_model=verifier_model,
        draft_model_path="/mnt/persistent/final_model",
        evaluation_dataset_uri=evaluation_dataset_uri,
        evaluation_dataset_pvc=persistent_pvc,
        evaluation_speculator_type=speculator_type,
        evaluation_max_samples=evaluation_max_samples,
        evaluation_max_tokens=evaluation_max_tokens,
        evaluation_temperature=evaluation_temperature,
        evaluation_num_speculative_tokens=evaluation_num_speculative_tokens,
        evaluation_gpu_memory_utilization=evaluation_gpu_memory_utilization,
        evaluation_startup_timeout_seconds=evaluation_startup_timeout_seconds,
        evaluation_sample_seed=evaluation_sample_seed,
    )
    evaluation_task.after(task)
    kfp.kubernetes.mount_pvc(
        evaluation_task,
        pvc_name=persistent_pvc,
        mount_path="/mnt/persistent",
    )
    evaluation_task.set_accelerator_type("nvidia.com/gpu")
    _attach_k8s(evaluation_task)

    model_registry_task = kubeflow_model_registry(
        pvc_mount_path="/mnt/persistent",
        input_model=task.outputs["output_model"],
        input_metrics=task.outputs["output_metrics"],
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
        verifier_model_dir=verifier_model,
        draft_model_dir="final_model",
        runtime_image=serving_runtime_image,
        max_model_len=total_seq_len,
    )
    serving_task.after(model_registry_task)
    serving_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(serving_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=speculator_offline_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )

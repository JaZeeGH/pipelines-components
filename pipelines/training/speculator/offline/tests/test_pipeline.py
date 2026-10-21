"""Tests for the Speculator offline pipeline."""

from kfp import compiler

from ..pipeline import speculator_offline_pipeline

PUBLIC_INPUTS = [
    "p01_persistent_pvc",
    "p02_verifier_model",
    "p03_verifier_model_pvc",
    "p04_download_model",
    "p05_dataset_name",
    "p06_dataset_download_enabled",
    "p07_downloaded_dataset_uri",
    "p08_data_extraction_max_samples",
    "p09_data_extraction_regenerate_responses",
    "p10_data_extraction_concurrency",
    "p11_vllm_resource_gpu",
    "p12_vllm_resource_memory",
    "p13_vllm_gpu_memory_utilization",
    "p14_vllm_readiness_timeout_minutes",
    "p15_extraction_runtime",
    "p16_output_dir",
    "p17_hidden_states_path",
    "p18_training_data_path",
    "p19_total_seq_len",
    "p20_speculator_type",
    "p21_target_layer_ids",
    "p22_training_epochs",
    "p23_training_lr",
    "p24_training_draft_vocab_size",
    "p25_training_num_layers",
    "p26_training_test_time_steps",
    "p27_training_norm_before_residual",
    "p28_training_norm_before_fc",
    "p29_training_embed_requires_grad",
    "p30_training_hidden_states_dtype",
    "p31_training_scheduler_type",
    "p32_training_scheduler_warmup_steps",
    "p33_training_scheduler_total_steps",
    "p34_training_scheduler_num_cosine_cycles",
    "p35_training_checkpoint_freq",
    "p36_training_save_best",
    "p37_training_log_freq",
    "p38_training_resume_from_checkpoint",
    "p39_training_resource_cpu",
    "p40_training_resource_gpu",
    "p41_training_resource_memory",
    "p42_job_envs",
    "p43_job_runtime",
    "p44_packages_to_install",
    "p45_pip_index_urls",
    "p46_evaluation_dataset_uri",
    "p47_evaluation_max_samples",
    "p48_evaluation_max_tokens",
    "p49_evaluation_temperature",
    "p50_evaluation_num_speculative_tokens",
    "p51_evaluation_gpu_memory_utilization",
    "p52_evaluation_startup_timeout_seconds",
    "p53_evaluation_sample_seed",
    "p54_model_registry_address",
    "p55_model_registry_port",
    "p56_model_registry_name",
    "p57_model_registry_version",
    "p58_serving_namespace",
    "p59_serving_endpoint_name",
    "p60_serving_runtime_image",
]


def _compile_pipeline(path):
    compiler.Compiler().compile(speculator_offline_pipeline, package_path=str(path))


def _load_pipeline_documents(path):
    import yaml

    return list(yaml.safe_load_all(path.read_text(encoding="utf-8")))


def _executor_config(pipeline_spec, platform_spec, task):
    component_name = pipeline_spec["root"]["dag"]["tasks"][task]["componentRef"]["name"]
    executor_label = pipeline_spec["components"][component_name]["executorLabel"]
    return platform_spec["platforms"]["kubernetes"]["deploymentSpec"]["executors"][executor_label]


def test_pipeline_exposes_stage_ordered_inputs():
    """Numeric parameter prefixes preserve the UI's stage-oriented input order."""
    inputs = speculator_offline_pipeline.component_spec.inputs

    assert list(inputs) == PUBLIC_INPUTS
    assert inputs["p01_persistent_pvc"].default is None
    assert inputs["p02_verifier_model"].default == ""
    assert inputs["p03_verifier_model_pvc"].default == ""
    assert inputs["p04_download_model"].default is False
    assert inputs["p06_dataset_download_enabled"].default is False
    assert inputs["p07_downloaded_dataset_uri"].default == ""
    assert inputs["p21_target_layer_ids"].default == ""
    assert inputs["p22_training_epochs"].default == 3
    assert inputs["p46_evaluation_dataset_uri"].default == ""
    assert inputs["p54_model_registry_address"].default == ""
    assert inputs["p58_serving_namespace"].default == ""
    assert inputs["p60_serving_runtime_image"].default == (
        "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9"
        "@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a"
    )
    assert inputs["p05_dataset_name"].default == "ultrachat"
    assert inputs["p08_data_extraction_max_samples"].default == 16
    assert inputs["p10_data_extraction_concurrency"].default == 1
    assert "training_from_pretrained" not in inputs


def test_pipeline_compiles_and_orders_all_stages(tmp_path):
    """Extraction, training, evaluation, registration, and serving run in order."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, _ = _load_pipeline_documents(output_path)

    assert "speculator-offline-pipeline" in output_path.read_text(encoding="utf-8")
    assert sorted(pipeline_spec["root"]["inputDefinitions"]["parameters"]) == PUBLIC_INPUTS

    tasks = pipeline_spec["root"]["dag"]["tasks"]
    assert "download-model" in tasks["extract-speculator"]["dependentTasks"]
    assert "extract-speculator" in tasks["train-speculator-mode"]["dependentTasks"]
    assert "train-speculator-mode" in tasks["evaluate-speculator"]["dependentTasks"]
    assert "evaluate-speculator" in tasks["kubeflow-model-registry"]["dependentTasks"]
    assert "kubeflow-model-registry" in tasks["speculative-decoding-serving"]["dependentTasks"]

    assert tasks["train-speculator-mode"]["taskInfo"]["name"] == "train-speculator-mode"
    assert tasks["evaluate-speculator"]["taskInfo"]["name"] == "evaluate-speculator"
    assert "displayName" not in tasks["train-speculator-mode"]["taskInfo"]
    assert "displayName" not in tasks["evaluate-speculator"]["taskInfo"]


def test_pipeline_wires_shared_pvc_paths_and_stage_inputs(tmp_path):
    """The extraction output, training input, evaluator, Registry, and serving share the PVC."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, _ = _load_pipeline_documents(output_path)
    tasks = pipeline_spec["root"]["dag"]["tasks"]

    download_inputs = tasks["download-model"]["inputs"]["parameters"]
    assert download_inputs["download_enabled"]["componentInputParameter"] == "p04_download_model"
    assert download_inputs["model_name"]["componentInputParameter"] == "p02_verifier_model"

    extraction_inputs = tasks["extract-speculator"]["inputs"]["parameters"]
    assert extraction_inputs["persistent_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert extraction_inputs["verifier_model_pvc"]["componentInputParameter"] == "p03_verifier_model_pvc"
    assert extraction_inputs["dataset_download_enabled"]["componentInputParameter"] == "p06_dataset_download_enabled"
    assert extraction_inputs["downloaded_dataset_uri"]["componentInputParameter"] == "p07_downloaded_dataset_uri"
    assert extraction_inputs["dataset_name"]["componentInputParameter"] == "p05_dataset_name"
    extraction_max_samples = extraction_inputs["data_extraction_max_samples"]
    assert extraction_max_samples["componentInputParameter"] == "p08_data_extraction_max_samples"

    training_inputs = tasks["train-speculator-mode"]["inputs"]["parameters"]
    assert training_inputs["persistent_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert training_inputs["verifier_model_pvc"]["componentInputParameter"] == "p03_verifier_model_pvc"
    assert training_inputs["training_data_path"]["componentInputParameter"] == "p18_training_data_path"
    assert training_inputs["hidden_states_path"]["componentInputParameter"] == "p17_hidden_states_path"
    assert training_inputs["training_ttt_steps"]["componentInputParameter"] == "p26_training_test_time_steps"

    evaluation_inputs = tasks["evaluate-speculator"]["inputs"]["parameters"]
    assert evaluation_inputs["evaluation_dataset_uri"]["componentInputParameter"] == "p46_evaluation_dataset_uri"
    assert evaluation_inputs["evaluation_dataset_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert (
        evaluation_inputs["evaluation_num_speculative_tokens"]["componentInputParameter"]
        == "p50_evaluation_num_speculative_tokens"
    )

    registry_inputs = tasks["kubeflow-model-registry"]["inputs"]["parameters"]
    assert registry_inputs["model_pvc_name"]["componentInputParameter"] == "p01_persistent_pvc"
    assert registry_inputs["registry_address"]["componentInputParameter"] == "p54_model_registry_address"
    assert registry_inputs["model_name"]["componentInputParameter"] == "p56_model_registry_name"
    assert registry_inputs["registry_port"]["componentInputParameter"] == "p55_model_registry_port"

    serving_inputs = tasks["speculative-decoding-serving"]["inputs"]["parameters"]
    assert serving_inputs["model_cache_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert serving_inputs["namespace"]["componentInputParameter"] == "p58_serving_namespace"
    assert serving_inputs["endpoint_name"]["componentInputParameter"] == "p59_serving_endpoint_name"
    assert serving_inputs["verifier_model_dir"]["componentInputParameter"] == "p03_verifier_model_pvc"
    assert serving_inputs["runtime_image"]["componentInputParameter"] == "p60_serving_runtime_image"


def test_pipeline_limits_kubernetes_credentials_to_trainer_tasks(tmp_path):
    """Evaluation and model-download tasks receive no Kubernetes credentials."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, platform_spec = _load_pipeline_documents(output_path)

    for task_name in ("extract-speculator", "train-speculator-mode"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name)["secretAsEnv"]
        assert {secret["secretName"] for secret in secrets} == {"kubernetes-credentials", "hf-token"}

    for task_name in ("download-model", "evaluate-speculator"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name)["secretAsEnv"]
        assert {secret["secretName"] for secret in secrets} == {"hf-token"}
        assert all(secret["optional"] is True for secret in secrets)

    for task_name in ("kubeflow-model-registry", "speculative-decoding-serving"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name).get("secretAsEnv", [])
        assert secrets == []

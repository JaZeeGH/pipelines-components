"""Tests for the Speculator offline pipeline."""

from kfp import compiler

from ..pipeline import speculator_offline_pipeline

PUBLIC_INPUTS = [
    "p01_persistent_pvc",
    "p02_verifier_model_pvc",
    "p03_vllm_endpoint",
    "p04_dataset_name",
    "p05_data_extraction_max_samples",
    "p06_data_extraction_regenerate_responses",
    "p07_data_extraction_concurrency",
    "p08_output_dir",
    "p09_hidden_states_path",
    "p10_training_data_path",
    "p11_total_seq_len",
    "p12_speculator_type",
    "p13_target_layer_ids",
    "p14_training_epochs",
    "p15_training_lr",
    "p16_training_draft_vocab_size",
    "p17_training_num_layers",
    "p18_training_test_time_steps",
    "p19_training_norm_before_residual",
    "p20_training_norm_before_fc",
    "p21_training_embed_requires_grad",
    "p22_training_hidden_states_dtype",
    "p23_training_scheduler_type",
    "p24_training_scheduler_warmup_steps",
    "p25_training_scheduler_total_steps",
    "p26_training_scheduler_num_cosine_cycles",
    "p27_training_checkpoint_freq",
    "p28_training_save_best",
    "p29_training_log_freq",
    "p30_training_resume_from_checkpoint",
    "p31_training_resource_cpu",
    "p32_training_resource_gpu",
    "p33_training_resource_memory",
    "p34_job_envs",
    "p35_job_runtime",
    "p36_packages_to_install",
    "p37_pip_index_urls",
    "p38_evaluation_dataset_uri",
    "p39_evaluation_max_samples",
    "p40_evaluation_max_tokens",
    "p41_evaluation_temperature",
    "p42_evaluation_num_speculative_tokens",
    "p43_evaluation_gpu_memory_utilization",
    "p44_evaluation_startup_timeout_seconds",
    "p45_evaluation_sample_seed",
    "p46_model_registry_address",
    "p47_model_registry_port",
    "p48_model_registry_name",
    "p49_model_registry_version",
    "p50_serving_namespace",
    "p51_serving_endpoint_name",
    "p52_serving_runtime_image",
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
    assert inputs["p02_verifier_model_pvc"].default is None
    assert inputs["p03_vllm_endpoint"].default is None
    assert inputs["p04_dataset_name"].default is None
    assert inputs["p05_data_extraction_max_samples"].default == 0
    assert inputs["p06_data_extraction_regenerate_responses"].default is False
    assert inputs["p07_data_extraction_concurrency"].default == 1
    assert inputs["p13_target_layer_ids"].default == ""
    assert inputs["p14_training_epochs"].default == 3
    assert inputs["p35_job_runtime"].default == "speculator-model-opt-cuda"
    assert inputs["p38_evaluation_dataset_uri"].default == ""
    assert inputs["p46_model_registry_address"].default == ""
    assert inputs["p50_serving_namespace"].default == ""
    assert inputs["p52_serving_runtime_image"].default == (
        "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9"
        "@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a"
    )
    assert "training_from_pretrained" not in inputs


def test_pipeline_compiles_and_orders_all_stages(tmp_path):
    """Extraction, training, evaluation, registration, and serving run in order."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, _ = _load_pipeline_documents(output_path)

    assert "speculator-offline-pipeline" in output_path.read_text(encoding="utf-8")
    assert sorted(pipeline_spec["root"]["inputDefinitions"]["parameters"]) == PUBLIC_INPUTS

    tasks = pipeline_spec["root"]["dag"]["tasks"]
    assert "download-model" not in tasks
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

    extraction_inputs = tasks["extract-speculator"]["inputs"]["parameters"]
    assert extraction_inputs["persistent_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert extraction_inputs["verifier_model_pvc"]["componentInputParameter"] == "p02_verifier_model_pvc"
    assert extraction_inputs["dataset_name"]["componentInputParameter"] == "p04_dataset_name"
    assert extraction_inputs["vllm_source"]["runtimeValue"]["constant"] == "remote"
    assert extraction_inputs["vllm_endpoint"]["componentInputParameter"] == "p03_vllm_endpoint"
    assert extraction_inputs["training_runtime"]["componentInputParameter"] == "p35_job_runtime"
    extraction_max_samples = extraction_inputs["data_extraction_max_samples"]
    assert extraction_max_samples["componentInputParameter"] == "p05_data_extraction_max_samples"
    assert extraction_inputs["hidden_states_path"]["componentInputParameter"] == "p09_hidden_states_path"
    assert (
        extraction_inputs["data_extraction_concurrency"]["componentInputParameter"] == "p07_data_extraction_concurrency"
    )

    training_inputs = tasks["train-speculator-mode"]["inputs"]["parameters"]
    assert training_inputs["persistent_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert training_inputs["verifier_model_pvc"]["componentInputParameter"] == "p02_verifier_model_pvc"
    assert training_inputs["training_data_path"]["componentInputParameter"] == "p10_training_data_path"
    assert training_inputs["hidden_states_path"]["componentInputParameter"] == "p09_hidden_states_path"
    assert training_inputs["training_ttt_steps"]["componentInputParameter"] == "p18_training_test_time_steps"
    assert training_inputs["training_runtime"]["componentInputParameter"] == "p35_job_runtime"

    evaluation_inputs = tasks["evaluate-speculator"]["inputs"]["parameters"]
    assert evaluation_inputs["evaluation_dataset_uri"]["componentInputParameter"] == "p38_evaluation_dataset_uri"
    assert evaluation_inputs["evaluation_dataset_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert (
        evaluation_inputs["evaluation_num_speculative_tokens"]["componentInputParameter"]
        == "p42_evaluation_num_speculative_tokens"
    )

    registry_inputs = tasks["kubeflow-model-registry"]["inputs"]["parameters"]
    assert registry_inputs["model_pvc_name"]["componentInputParameter"] == "p01_persistent_pvc"
    assert registry_inputs["registry_address"]["componentInputParameter"] == "p46_model_registry_address"
    assert registry_inputs["model_name"]["componentInputParameter"] == "p48_model_registry_name"
    assert registry_inputs["registry_port"]["componentInputParameter"] == "p47_model_registry_port"

    serving_inputs = tasks["speculative-decoding-serving"]["inputs"]["parameters"]
    assert serving_inputs["model_cache_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert serving_inputs["namespace"]["componentInputParameter"] == "p50_serving_namespace"
    assert serving_inputs["endpoint_name"]["componentInputParameter"] == "p51_serving_endpoint_name"
    assert serving_inputs["verifier_model_dir"]["componentInputParameter"] == "p02_verifier_model_pvc"
    assert serving_inputs["runtime_image"]["componentInputParameter"] == "p52_serving_runtime_image"


def test_pipeline_limits_kubernetes_credentials_to_trainer_tasks(tmp_path):
    """Evaluation does not receive Kubernetes Trainer credentials."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, platform_spec = _load_pipeline_documents(output_path)

    for task_name in ("extract-speculator", "train-speculator-mode"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name)["secretAsEnv"]
        assert {secret["secretName"] for secret in secrets} == {"kubernetes-credentials", "hf-token"}

    secrets = _executor_config(pipeline_spec, platform_spec, "evaluate-speculator")["secretAsEnv"]
    assert {secret["secretName"] for secret in secrets} == {"hf-token"}
    assert all(secret["optional"] is True for secret in secrets)

    for task_name in ("kubeflow-model-registry", "speculative-decoding-serving"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name).get("secretAsEnv", [])
        assert secrets == []

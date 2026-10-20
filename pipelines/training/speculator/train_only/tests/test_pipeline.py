"""Tests for the Speculator train-only pipeline."""

import yaml
from kfp import compiler

from ..pipeline import speculator_training_pipeline


def test_pipeline_exposes_training_registry_and_serving_inputs():
    """The pipeline exposes its PVC, training, Registry, and serving controls."""
    inputs = speculator_training_pipeline.component_spec.inputs
    expected = {
        "persistent_pvc",
        "hidden_states_path",
        "training_data_path",
        "model_registry_address",
        "model_registry_port",
        "serving_namespace",
        "serving_runtime_image",
        "serving_endpoint_name",
        "verifier_model_pvc",
    }
    assert expected <= set(inputs)
    assert "verifier_model" not in inputs
    assert "download_model" not in inputs
    assert inputs["target_layer_ids"].default == "2,14,25,28"
    assert inputs["training_data_path"].default == "speculator-output"
    assert inputs["evaluation_dataset_uri"].default == ""
    assert inputs["model_registry_name"].default is None
    assert inputs["serving_endpoint_name"].default is None
    assert inputs["verifier_model_pvc"].default == ""
    assert inputs["job_envs"].default == "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1"
    assert inputs["job_runtime"].default == "speculator-model-opt-cuda"
    assert "training_envs" not in inputs
    assert "training_runtime" not in inputs


def test_progression_metrics_are_not_public_controls():
    """Progress tracking uses the training component's always-on defaults."""
    inputs = speculator_training_pipeline.component_spec.inputs

    assert "enable_progression_tracking" not in inputs
    assert "metrics_port" not in inputs
    assert "metrics_poll_interval_seconds" not in inputs


def test_pipeline_compiles_and_orders_training_evaluation_registry_serving(tmp_path):
    """Evaluation follows training, registration follows evaluation, then serving."""
    output_path = tmp_path / "speculator_training.yaml"
    compiler.Compiler().compile(speculator_training_pipeline, package_path=str(output_path))

    compiled_pipeline = output_path.read_text()
    assert "speculator-training-pipeline" in compiled_pipeline
    pipeline_spec = next(yaml.safe_load_all(compiled_pipeline))
    tasks = pipeline_spec["root"]["dag"]["tasks"]
    assert "model-registry==0.3.9" in compiled_pipeline
    assert "model-registry==0.3.4" not in compiled_pipeline
    assert "train-speculator-mode" in tasks["evaluate-speculator"]["dependentTasks"]
    assert "evaluate-speculator" in tasks["kubeflow-model-registry"]["dependentTasks"]
    assert "kubeflow-model-registry" in tasks["speculative-decoding-serving"]["dependentTasks"]

    train_inputs = tasks["train-speculator-mode"]["inputs"]["parameters"]
    assert train_inputs["persistent_pvc"]["componentInputParameter"] == "persistent_pvc"
    assert train_inputs["hidden_states_path"]["componentInputParameter"] == "hidden_states_path"
    assert train_inputs["training_data_path"]["componentInputParameter"] == "training_data_path"

    evaluation_inputs = tasks["evaluate-speculator"]["inputs"]["parameters"]
    assert evaluation_inputs["evaluation_dataset_pvc"]["componentInputParameter"] == "persistent_pvc"

    registry_inputs = tasks["kubeflow-model-registry"]["inputs"]["parameters"]
    assert registry_inputs["model_pvc_name"]["componentInputParameter"] == "persistent_pvc"

    serving_inputs = tasks["speculative-decoding-serving"]["inputs"]["parameters"]
    assert serving_inputs["model_cache_pvc"]["componentInputParameter"] == "persistent_pvc"
    assert serving_inputs["verifier_model_dir"]["componentInputParameter"] == "verifier_model_pvc"

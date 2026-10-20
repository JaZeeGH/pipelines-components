"""Tests for the managed Speculator data-only pipeline."""

from kfp import compiler

from ..pipeline import speculator_hidden_state_extraction_pipeline


class TestSpeculatorHiddenStateExtractionPipeline:
    """Basic tests for the data-only pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is defined."""
        assert callable(speculator_hidden_state_extraction_pipeline)

    def test_verifier_model_inputs_do_not_assume_cached_models(self):
        """The pipeline does not assume a model was previously downloaded to the PVC."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["verifier_model"].default == ""
        assert inputs["verifier_model_pvc"].default == ""

    def test_vllm_mode_is_the_public_control(self):
        """The pipeline exposes the deployment mode, not an ambiguous source name."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["vllm_mode"].default == "managed"
        assert "vllm_source" not in inputs

    def test_extraction_controls_have_concise_public_names(self):
        """Extraction controls use names appropriate for the pipeline UI."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["max_samples"].default == 16
        assert inputs["regenerate_responses"].default is False
        assert "data_extraction_max_samples" not in inputs
        assert "data_extraction_regenerate_responses" not in inputs

    def test_job_runtime_controls_have_clear_public_names(self):
        """The pipeline exposes job controls without Trainer-specific prefixes."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["job_envs"].default == "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1"
        assert inputs["job_runtime"].default == "vllm-extract-cuda"
        assert "training_envs" not in inputs
        assert "training_runtime" not in inputs

    def test_progression_metrics_are_not_public_controls(self):
        """Progress tracking uses the extraction component's defaults."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert "enable_progression_tracking" not in inputs
        assert "metrics_port" not in inputs
        assert "metrics_poll_interval_seconds" not in inputs

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=speculator_hidden_state_extraction_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        compiled = output_path.read_text()
        assert "speculator-hidden-state-extraction-pipeline" in compiled
        assert "verifier hidden states for speculative decoding training" in compiled
        assert "managed-vllm" in compiled
        assert "remote-vllm" in compiled
        assert "extract-hidden-states" in compiled
        assert "vllm-connected" not in compiled
        assert "vllm-disconnected" not in compiled
        assert "extract-speculator" not in compiled

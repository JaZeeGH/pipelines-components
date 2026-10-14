"""Tests for the Speculator offline pipeline."""

from kfp import compiler

from ..pipeline import speculator_offline_pipeline


class TestSpeculatorOfflinePipeline:
    """Basic tests for the offline pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is defined."""
        assert callable(speculator_offline_pipeline)

    def test_pipeline_exposes_serving_inputs(self):
        """The offline pipeline exposes evaluation and serving inputs."""
        inputs = speculator_offline_pipeline.component_spec.inputs
        for name in (
            "evaluation_dataset_uri",
            "evaluation_max_samples",
            "evaluation_num_speculative_tokens",
            "serving_namespace",
            "serving_runtime_image",
            "serving_endpoint_name",
        ):
            assert name in inputs

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=speculator_offline_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()

    def test_serving_runs_after_evaluation(self, tmp_path):
        """Serving is scheduled after the offline evaluation task completes."""
        import yaml

        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=speculator_offline_pipeline,
            package_path=str(output_path),
        )
        pipeline_spec = next(yaml.safe_load_all(output_path.read_text()))
        tasks = pipeline_spec["root"]["dag"]["tasks"]

        assert "evaluate-speculator" in tasks["speculative-decoding-serving"]["dependentTasks"]

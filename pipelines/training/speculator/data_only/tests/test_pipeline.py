"""Tests for the managed Speculator data-only pipeline."""

from kfp import compiler

from ..pipeline import speculator_data_only_pipeline


class TestSpeculatorDataOnlyPipeline:
    """Basic tests for the data-only pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is defined."""
        assert callable(speculator_data_only_pipeline)

    def test_verifier_model_defaults_empty_for_cached_model_reuse(self):
        """A pre-downloaded verifier path can be used without a competing HF default."""
        inputs = speculator_data_only_pipeline.component_spec.inputs

        assert inputs["verifier_model"].default == ""

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=speculator_data_only_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()

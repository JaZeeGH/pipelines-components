"""Unit tests for the Speculator extraction component."""

import inspect

import pytest

from ..component import _resolve_verifier_model_pvc_uri, extract_speculator


class TestSpeculatorExtractionComponent:
    """Tests for the fixed extraction component interface."""

    def test_component_function_exists(self):
        """Verify the extraction component is exposed as a KFP component."""
        assert callable(extract_speculator)
        assert hasattr(extract_speculator, "python_func")

    def test_component_has_extraction_parameters(self):
        """Verify extraction-specific parameters are exposed."""
        params = inspect.signature(extract_speculator.python_func).parameters
        for name in ("verifier_model", "dataset_name", "output_hidden_states", "vllm_endpoint"):
            assert name in params

    def test_component_has_fixed_mode(self):
        """Verify extraction does not expose a mode selector."""
        assert "mode" not in inspect.signature(extract_speculator.python_func).parameters

    def test_remote_verifier_relative_path_resolves_to_pvc_uri(self):
        """A relative PVC verifier path becomes an explicit Trainer URI for remote vLLM."""
        assert _resolve_verifier_model_pvc_uri("models/Qwen--Qwen3-0.6B", "shared") == (
            "pvc://shared/models/Qwen--Qwen3-0.6B"
        )

    def test_remote_verifier_uri_must_reference_the_mounted_pvc(self):
        """An explicit verifier URI cannot point to a different PVC."""
        with pytest.raises(ValueError, match="must reference persistent_pvc"):
            _resolve_verifier_model_pvc_uri("pvc://other/models/verifier", "shared")

    @pytest.mark.parametrize("path", ("", "../models/verifier", "/models/verifier"))
    def test_remote_verifier_path_rejects_empty_or_unsafe_paths(self, path):
        """Remote verifier paths must be non-empty relative paths without traversal."""
        with pytest.raises(ValueError):
            _resolve_verifier_model_pvc_uri(path, "shared")

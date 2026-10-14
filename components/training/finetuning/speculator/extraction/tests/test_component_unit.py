"""Unit tests for the Speculator extraction component."""

import inspect
import sys
from unittest import mock

from ..component import extract_speculator


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

    def test_managed_mode_uses_pvc_verifier_when_download_is_disabled(self, monkeypatch):
        """Managed extraction uses the configured PVC model when reusing a cached verifier."""
        run_speculator = mock.Mock(return_value="ok")
        fake_speculator = mock.MagicMock(run_speculator=run_speculator)
        monkeypatch.setitem(sys.modules, "speculator", fake_speculator)

        result = extract_speculator.python_func(
            verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
            verifier_model_pvc="models/Qwen3-0.6B",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="ultrachat",
            vllm_source="managed",
            download_model=False,
        )

        assert result == "ok"
        values = run_speculator.call_args.args[1]
        assert values["verifier_model"] == "models/Qwen3-0.6B"

    def test_managed_mode_keeps_hf_verifier_when_download_is_enabled(self, monkeypatch):
        """Managed extraction keeps the HF verifier when the pipeline will download it."""
        run_speculator = mock.Mock(return_value="ok")
        monkeypatch.setitem(
            sys.modules,
            "speculator",
            mock.MagicMock(run_speculator=run_speculator),
        )

        extract_speculator.python_func(
            verifier_model="Qwen/Qwen3-0.6B",
            verifier_model_pvc="models/Qwen--Qwen3-0.6B",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="ultrachat",
            vllm_source="managed",
            download_model=True,
        )

        values = run_speculator.call_args.args[1]
        assert values["verifier_model"] == "Qwen/Qwen3-0.6B"

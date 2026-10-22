"""Speculator Training Component."""

from .extraction import extract_hidden_states
from .online import online_speculator
from .training import train_draft_model

__all__ = ["extract_hidden_states", "online_speculator", "train_draft_model"]

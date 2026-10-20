"""Speculator Training Component."""

from .extraction import extract_hidden_states
from .online import online_speculator
from .training import train_speculator_mode

__all__ = ["extract_hidden_states", "online_speculator", "train_speculator_mode"]

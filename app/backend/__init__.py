"""Shared backend for OSS + frontier personal assistants."""
from .oss_assistant import OSSAssistant
from .frontier_assistant import FrontierAssistant

__all__ = ["OSSAssistant", "FrontierAssistant"]

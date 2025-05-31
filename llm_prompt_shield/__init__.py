"""
LLM Prompt Shield - Lightweight prompt injection detection and blocking
"""

from .detector import PromptGuard
from .user_config import init_user_config, edit_config

# Simple API functions
def is_safe(prompt: str, config=None) -> bool:
    """Quick safety check for a prompt."""
    guard = PromptGuard()
    try:
        result = guard.analyze_sync(prompt, config)
        return result['action'] != 'block'
    finally:
        guard.close()

def analyze(prompt: str, config=None) -> dict:
    """Detailed analysis of a prompt."""
    guard = PromptGuard()
    try:
        return guard.analyze_sync(prompt, config)
    finally:
        guard.close()

__version__ = "0.1.1"
__all__ = [
    "PromptGuard",
    "is_safe",
    "analyze",
    "init_user_config",
    "edit_config"
]
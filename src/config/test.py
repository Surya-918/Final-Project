"""Compatibility wrapper for the old backend module name.

Use src.config.app as the canonical entry point. This file remains only to avoid
breaking older scripts and documentation while the project transitions.
"""

from .app import *  # noqa: F401,F403

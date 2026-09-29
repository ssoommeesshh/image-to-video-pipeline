"""Notebook and CLI interface for CPU educational-video integration."""
from .pipeline import VideoPipeline, RunResult
from .catalog import Catalog
from .contract import load_plan, validate_plan

__all__ = ["VideoPipeline", "RunResult", "Catalog", "load_plan", "validate_plan"]

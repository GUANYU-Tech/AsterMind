"""Compatibility import for existing Cell callers."""

from .ArchPrism.scheduler.serving import ServingScheduler as MidlayerCell

__all__ = ["MidlayerCell"]

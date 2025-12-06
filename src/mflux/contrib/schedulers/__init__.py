"""
mflux-schedulers: Advanced Schedulers for mflux

This package provides additional schedulers for mflux image generation models.
All schedulers are compatible with FLUX, Qwen, Z-Image, and FIBO models.

Installation:
    pip install mflux-schedulers

Usage:
    # In CLI:
    mflux-generate --scheduler mflux.contrib.schedulers.ddim --steps 10

    # In code:
    from mflux import Flux1, Config
    flux = Flux1.from_name("schnell")
    config = Config(scheduler="mflux.contrib.schedulers.ddim")
    image = flux.generate_image(seed=42, prompt="a cat", config=config)

Available Schedulers:
    - ddim: DDIM Flow Matching (10x faster, great quality)
    - stork: STORK Stabilized RK (excellent quality)
    - stork-2: STORK Order-2 (faster)
    - stork-4: STORK Order-4 (highest quality)
    - er_sde_beta: ER-SDE with Beta sampling (best details)
    - advanced: Multiple noise schedules (experimental)
"""

from .base_scheduler import BaseScheduler
from .scheduler_factory import SchedulerFactory
from .scheduler_metadata import (
    SCHEDULER_METADATA,
    SchedulerMetadata,
    SchedulerStatus,
    register_metadata,
)

# Import all built-in schedulers
from .builtin import DDIMFlowScheduler, STORKScheduler

__version__ = "0.1.0"

# Auto-register built-in schedulers
SchedulerFactory.register("ddim", DDIMFlowScheduler)
SchedulerFactory.register("stork", STORKScheduler)  # Default to order 2

# Factory functions for STORK variants
SchedulerFactory.register("stork-2", lambda config, **kw: STORKScheduler(config, order=2, **kw))
SchedulerFactory.register("stork-4", lambda config, **kw: STORKScheduler(config, order=4, **kw))

# Aliases for convenience
SchedulerFactory.register("DDIMFlowScheduler", DDIMFlowScheduler)
SchedulerFactory.register("STORKScheduler", STORKScheduler)

__all__ = [
    # Core classes
    "BaseScheduler",
    "SchedulerFactory",
    "SchedulerMetadata",
    "SchedulerStatus",
    # Built-in schedulers
    "DDIMFlowScheduler",
    "STORKScheduler",
    # Registry
    "SCHEDULER_METADATA",
    "register_metadata",
    # Version
    "__version__",
]


def list_schedulers(status: str | None = None) -> list[str]:
    """
    List available schedulers.

    Args:
        status: Filter by status ('stable', 'beta', 'experimental', 'deprecated')

    Returns:
        list[str]: List of scheduler names

    Example:
        >>> import mflux.contrib.schedulers as schedulers
        >>> schedulers.list_schedulers()
        ['ddim']
        >>> schedulers.list_schedulers('stable')
        ['ddim']
    """
    if status:
        status_enum = SchedulerStatus(status)
        return SchedulerFactory.list_available(status_filter=status_enum)
    return SchedulerFactory.list_available()


def get_scheduler_info(name: str) -> dict:
    """
    Get detailed information about a scheduler.

    Args:
        name: Scheduler name

    Returns:
        dict: Scheduler metadata

    Example:
        >>> import mflux.contrib.schedulers as schedulers
        >>> info = schedulers.get_scheduler_info('ddim')
        >>> print(info['description'])
        'True DDIM-style accelerated sampling with quadratic spacing'
    """
    return SchedulerFactory.get_info(name)

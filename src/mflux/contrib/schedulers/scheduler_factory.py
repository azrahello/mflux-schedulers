"""
Scheduler Factory

Provides a factory pattern for creating scheduler instances with validation,
deprecation warnings, and metadata support.
"""

import warnings
from typing import Callable, Type

from .base_scheduler import BaseScheduler
from .scheduler_metadata import SCHEDULER_METADATA, SchedulerStatus


class SchedulerFactory:
    """
    Factory for creating and managing scheduler instances.

    Features:
    - Automatic validation of scheduler parameters
    - Deprecation warnings for old schedulers
    - Metadata-based compatibility checking
    - Support for factory functions (e.g., stork-2, stork-4)
    """

    _registry: dict[str, Type[BaseScheduler] | Callable] = {}

    @classmethod
    def register(
        cls,
        name: str,
        scheduler_class: Type[BaseScheduler] | Callable,
        *,
        override: bool = False,
    ) -> None:
        """
        Register a scheduler class or factory function.

        Args:
            name: Scheduler identifier (used in CLI and code)
            scheduler_class: Scheduler class or factory function
            override: Allow overriding existing registration (default: False)

        Raises:
            ValueError: If name already registered and override=False

        Example:
            >>> SchedulerFactory.register("ddim", DDIMFlowScheduler)
            >>> # Or with a factory function:
            >>> SchedulerFactory.register(
            ...     "stork-2",
            ...     lambda config, **kw: STORKScheduler(config, order=2, **kw)
            ... )
        """
        if name in cls._registry and not override:
            raise ValueError(
                f"Scheduler '{name}' is already registered. "
                f"Use override=True to replace it."
            )

        cls._registry[name] = scheduler_class

    @classmethod
    def create(cls, name: str, config, **kwargs) -> BaseScheduler:
        """
        Create a scheduler instance with validation.

        Args:
            name: Scheduler name (must be registered)
            config: Config object (from mflux)
            **kwargs: Additional scheduler-specific parameters

        Returns:
            BaseScheduler: Configured scheduler instance

        Raises:
            ValueError: If scheduler not found or invalid parameters

        Example:
            >>> config = Config(...)
            >>> scheduler = SchedulerFactory.create("ddim", config, eta=0.0)
        """
        # Check if scheduler exists
        if name not in cls._registry:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"Scheduler '{name}' not found.\n"
                f"Available schedulers: {available}\n\n"
                f"Install with: pip install mflux-schedulers\n"
                f"Use with: --scheduler mflux.contrib.schedulers.{name}"
            )

        # Check status and emit warnings if needed
        metadata = SCHEDULER_METADATA.get(name)
        if metadata:
            if metadata.status == SchedulerStatus.DEPRECATED:
                warnings.warn(
                    f"Scheduler '{name}' is deprecated and will be removed "
                    f"in a future version. Please migrate to a supported scheduler.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            elif metadata.status == SchedulerStatus.EXPERIMENTAL:
                warnings.warn(
                    f"Scheduler '{name}' is experimental and may have bugs or change behavior. "
                    f"Use with caution in production.",
                    UserWarning,
                    stacklevel=2,
                )

        # Create instance
        scheduler_class = cls._registry[name]
        try:
            return scheduler_class(config, **kwargs)
        except TypeError as e:
            # Provide helpful error message
            if metadata:
                raise ValueError(
                    f"Failed to create scheduler '{name}': {e}\n\n"
                    f"Typical usage: {metadata.display_name}\n"
                    f"Steps: {metadata.typical_steps[0]}-{metadata.typical_steps[1]}\n"
                    f"Check your parameters: {kwargs}"
                ) from e
            else:
                raise ValueError(f"Failed to create scheduler '{name}': {e}") from e

    @classmethod
    def list_available(cls, status_filter: SchedulerStatus | None = None) -> list[str]:
        """
        List available schedulers, optionally filtered by status.

        Args:
            status_filter: Only include schedulers with this status (optional)

        Returns:
            list[str]: List of scheduler names

        Example:
            >>> SchedulerFactory.list_available()
            ['ddim', 'stork', 'stork-2', 'stork-4', 'er_sde_beta', 'advanced']
            >>> SchedulerFactory.list_available(SchedulerStatus.STABLE)
            ['ddim', 'er_sde_beta']
        """
        if status_filter is None:
            return sorted(cls._registry.keys())

        filtered = []
        for name in cls._registry.keys():
            metadata = SCHEDULER_METADATA.get(name)
            if metadata and metadata.status == status_filter:
                filtered.append(name)

        return sorted(filtered)

    @classmethod
    def get_info(cls, name: str) -> dict:
        """
        Get detailed information about a scheduler.

        Args:
            name: Scheduler name

        Returns:
            dict: Scheduler metadata and information

        Raises:
            ValueError: If scheduler not found or has no metadata

        Example:
            >>> info = SchedulerFactory.get_info("ddim")
            >>> print(info["description"])
            'True DDIM-style accelerated sampling with quadratic spacing'
        """
        if name not in cls._registry:
            raise ValueError(f"Scheduler '{name}' not found")

        if name not in SCHEDULER_METADATA:
            return {
                "name": name,
                "registered": True,
                "metadata_available": False,
            }

        return SCHEDULER_METADATA[name].to_dict()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a scheduler is registered."""
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a scheduler.

        Args:
            name: Scheduler name to remove

        Raises:
            KeyError: If scheduler not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Scheduler '{name}' is not registered")

        del cls._registry[name]

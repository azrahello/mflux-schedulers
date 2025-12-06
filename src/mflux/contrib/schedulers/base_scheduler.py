"""
Base Scheduler for mflux-schedulers

This module provides the base scheduler interface compatible with mflux.
All schedulers must inherit from BaseScheduler and implement the required methods.
"""

from abc import ABC, abstractmethod

import mlx.core as mx


class BaseScheduler(ABC):
    """
    Abstract base class for all schedulers.

    All schedulers must implement:
    - sigmas property: Returns the sigma schedule as mx.array
    - step method: Performs one denoising step
    - scale_model_input method (optional): Scales model input if needed
    """

    @property
    @abstractmethod
    def sigmas(self) -> mx.array:
        """
        Return the sigma schedule for the diffusion/flow process.

        Sigmas typically go from 1.0 (pure noise) to 0.0 (clean data).
        Should include an extra 0.0 at the end for the final step.

        Returns:
            mx.array: Sigma values of shape (num_inference_steps + 1,)
        """
        ...

    @abstractmethod
    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        """
        Perform one denoising step.

        Args:
            noise: Predicted noise/velocity from the model
            timestep: Current timestep index (0 to num_inference_steps - 1)
            latents: Current latent state
            **kwargs: Additional scheduler-specific parameters

        Returns:
            mx.array: Updated latents for the next timestep
        """
        ...

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the model input if needed by the scheduler.

        Most Flow Matching schedulers don't need scaling, so this defaults
        to returning the latents unchanged.

        Args:
            latents: Input latents
            t: Current timestep index

        Returns:
            mx.array: Scaled latents (or unchanged)
        """
        return latents

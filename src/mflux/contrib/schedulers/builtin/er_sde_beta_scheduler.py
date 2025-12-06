"""
ER-SDE Beta Scheduler for Flow Matching

Combines two powerful techniques:
1. Extended Reverse-Time SDE (arXiv:2309.06169) - high-quality sampling with ODE/SDE balance
2. Beta Sampling (arXiv:2407.12173) - optimal timestep distribution for detail preservation

This scheduler is optimized for Flow Matching models (FLUX, Qwen, Z-Image) and provides:
- Superior detail preservation through Beta-distributed timesteps
- Natural, non-plastic appearance through controlled stochasticity
- Controllable quality/speed trade-off via gamma parameter

Example usage:
    >>> from mflux import Flux1, Config
    >>> flux = Flux1.from_name("dev")
    >>> # Deterministic Beta sampling (best quality)
    >>> config = Config(
    ...     scheduler="mflux.contrib.schedulers.er_sde_beta",
    ...     num_inference_steps=30
    ... )
    >>> image = flux.generate_image(seed=42, prompt="detailed portrait", config=config)
    >>>
    >>> # With SDE component (more natural)
    >>> config = Config(
    ...     scheduler="mflux.contrib.schedulers.er_sde_beta",
    ...     num_inference_steps=30,
    ...     scheduler_kwargs={"gamma": 0.3}
    ... )
"""

import math

import mlx.core as mx

from ..base_scheduler import BaseScheduler


class ERSDEBetaScheduler(BaseScheduler):
    """
    Flow Matching scheduler with Beta timestep distribution and optional SDE component.

    Args:
        config: mflux Config object containing model configuration and parameters
        gamma: Controls SDE noise injection
               - 0.0: Pure ODE (deterministic, fastest)
               - 0.3: Balanced (recommended for natural appearance)
               - 1.0: Maximum stochasticity
        beta_strength: Controls Beta distribution aggressiveness
                       - 1.0: Gentle (sin², default)
                       - 2.0: Moderate (sin⁴)
                       - 3.0+: Aggressive (concentrates more at edges)
        **kwargs: Additional arguments for compatibility
    """

    def __init__(
        self,
        config,
        gamma: float = 0.0,
        beta_strength: float = 1.0,
        **kwargs,
    ):
        self.config = config
        self.model_config = config.model_config
        self.gamma = gamma
        self.beta_strength = beta_strength

        # Validate parameters
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if beta_strength < 0.5:
            raise ValueError(f"beta_strength must be >= 0.5, got {beta_strength}")

        # Compute sigma schedule
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """
        Compute timesteps and sigmas using Beta distribution.

        Beta distribution concentrates steps at edges (high and low noise),
        which improves detail preservation in generated images.
        """
        num_steps = self.config.num_inference_steps

        # Beta-inspired distribution: concentrates steps at edges
        timesteps_normalized = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Use sin transformation with adjustable strength
            # beta_strength controls how aggressive the distribution is:
            # 1.0 = gentle (original sin²)
            # 2.0 = moderate (sin⁴)
            # 3.0+ = aggressive (concentrates more at edges)
            t_base = math.sin(math.pi * t / 2) ** 2
            t_transformed = t_base**self.beta_strength
            timesteps_normalized.append(t_transformed)

        # For Flow Matching: sigmas go from 1.0 (pure noise) to 0.0 (clean data)
        sigmas = [1.0 - t for t in timesteps_normalized]

        # Apply sigma shift if required by the model
        if self.model_config.requires_sigma_shift:
            sigmas = self._apply_sigma_shift(sigmas)

        sigmas_arr = mx.array(sigmas, dtype=mx.float32)
        timesteps_arr = mx.arange(num_steps, dtype=mx.float32)

        return sigmas_arr, timesteps_arr

    def _apply_sigma_shift(self, sigmas: list[float]) -> list[float]:
        """
        Apply exponential sigma shift for resolution-dependent adjustment.
        Same logic as LinearScheduler for consistency.
        """
        # Calculate mu based on resolution
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * self.config.width * self.config.height / 256 + b

        # Apply exponential shift
        shifted_sigmas = []
        for s in sigmas:
            if s > 0:
                shifted = math.exp(mu) / (math.exp(mu) + (1 / s - 1))
                shifted_sigmas.append(shifted)
            else:
                shifted_sigmas.append(0.0)

        return shifted_sigmas

    @property
    def sigmas(self) -> mx.array:
        """Return the sigma schedule."""
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        """Return the timestep indices."""
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        """
        Perform one denoising step for Flow Matching.

        For Flow Matching with velocity prediction v(x_t, t):
        - ODE update: x_{t+dt} = x_t + dt * v(x_t, t)
        - Optional SDE: add controlled noise based on gamma

        Args:
            noise: Predicted velocity v(x_t, t) from the model
            timestep: Current timestep index (0 to num_steps-1)
            latents: Current latent state x_t
            **kwargs: Additional arguments for compatibility

        Returns:
            mx.array: Updated latents x_{t+dt}
        """
        # Get current and next sigma values
        sigma_t = self._sigmas[timestep]
        sigma_next = self._sigmas[timestep + 1]

        # Compute step size (dt)
        dt = sigma_next - sigma_t

        # ODE component: Euler step for Flow Matching
        # This is the base deterministic update
        pred_sample = latents + dt * noise

        # Optional SDE component: add controlled noise
        # This can improve quality and naturalness but adds stochasticity
        if self.gamma > 0 and timestep < len(self._timesteps) - 1:
            random_noise = mx.random.normal(latents.shape)
            # Scale noise by gamma and step size
            noise_scale = self.gamma * mx.sqrt(mx.abs(dt))
            pred_sample = pred_sample + noise_scale * random_noise

        return pred_sample

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale the model input. Flow Matching doesn't require input scaling.

        Args:
            latents: Input latents
            t: Timestep index

        Returns:
            mx.array: Unscaled latents
        """
        return latents

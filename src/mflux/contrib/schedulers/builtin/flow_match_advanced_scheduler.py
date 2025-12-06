"""
Flow Matching Advanced Scheduler with Multiple Noise Schedules

Supports various noise schedules for Flow Matching models:
- Linear, Cosine, Exponential, Sqrt, Scaled Linear (simple, effective)
- Beta distribution (complex, from arXiv:2407.12173)

Based on:
- DDPM noise schedules (Cosine, Exponential, Sqrt, Scaled Linear)
- Beta Sampling (arXiv:2407.12173, RES4LYF ComfyUI)

This scheduler separates NOISE SCHEDULE from SAMPLING METHOD,
allowing different timestep distributions with Euler integration.

Example usage:
    >>> from mflux import Flux1, Config
    >>> flux = Flux1.from_name("dev")
    >>> # Cosine (smooth, good perceptual quality)
    >>> config = Config(
    ...     scheduler="mflux.contrib.schedulers.advanced",
    ...     scheduler_kwargs={"schedule": "cosine"},
    ...     num_inference_steps=30
    ... )
    >>> image = flux.generate_image(seed=42, prompt="detailed portrait", config=config)
    >>>
    >>> # Beta RES_2M (from ComfyUI/Reddit)
    >>> config = Config(
    ...     scheduler="mflux.contrib.schedulers.advanced",
    ...     scheduler_kwargs={"schedule": "beta", "beta_alpha": 2.0, "beta_beta": 1.0},
    ...     num_inference_steps=30
    ... )
"""

import math
from typing import Literal

import mlx.core as mx

from ..base_scheduler import BaseScheduler


class FlowMatchAdvancedScheduler(BaseScheduler):
    """
    Flow Matching scheduler with multiple noise schedule options.

    Separates noise schedule from sampling method (Euler integration).

    Args:
        config: mflux Config object containing model configuration and parameters
        schedule: Noise schedule type
                 - "linear": Uniform spacing (baseline, fast)
                 - "cosine": Smoother transitions, better perceptual quality
                 - "exponential": Faster early denoising, refined details at end
                 - "sqrt": Preserves structure, good detail in complex areas
                 - "scaled_linear": Adaptive scaling for different image types
                 - "beta": Beta distribution (complex, concentrates steps at edges)
        exponential_beta: Beta parameter for exponential schedule (default: 2.0, range: 1.0-3.0 recommended)
        beta_alpha: Alpha parameter for beta schedule (default: 0.6)
        beta_beta: Beta parameter for beta schedule (default: 0.6)
        **kwargs: Additional arguments for compatibility

    Examples:
        # Cosine (smooth, good perceptual quality)
        --scheduler advanced --scheduler-kwargs '{"schedule": "cosine"}'

        # Exponential (fast early, refined end)
        --scheduler advanced --scheduler-kwargs '{"schedule": "exponential"}'

        # Sqrt (structure preservation)
        --scheduler advanced --scheduler-kwargs '{"schedule": "sqrt"}'

        # Beta RES_2M (from ComfyUI/Reddit)
        --scheduler advanced --scheduler-kwargs '{"schedule": "beta", "beta_alpha": 2.0, "beta_beta": 1.0}'
    """

    def __init__(
        self,
        config,
        schedule: Literal["linear", "cosine", "exponential", "sqrt", "scaled_linear", "beta"] = "linear",
        exponential_beta: float = 2.0,
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        **kwargs,
    ):
        self.config = config
        self.model_config = config.model_config
        self.schedule = schedule
        self.exponential_beta = exponential_beta
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

        # Compute sigma schedule
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """Compute timesteps and sigmas using selected distribution."""
        num_steps = self.config.num_inference_steps

        # Generate normalized timesteps [0, 1] using selected schedule
        if self.schedule == "beta":
            timesteps_normalized = self._beta_schedule(num_steps)
        elif self.schedule == "cosine":
            timesteps_normalized = self._cosine_schedule(num_steps)
        elif self.schedule == "exponential":
            timesteps_normalized = self._exponential_schedule(num_steps)
        elif self.schedule == "sqrt":
            timesteps_normalized = self._sqrt_schedule(num_steps)
        elif self.schedule == "scaled_linear":
            timesteps_normalized = self._scaled_linear_schedule(num_steps)
        else:  # linear
            timesteps_normalized = self._linear_schedule(num_steps)

        # Convert to sigmas for Flow Matching
        # sigmas go from 1.0 (pure noise) to 0.0 (clean data)
        sigmas = [1.0 - t for t in timesteps_normalized]

        # Apply sigma shift if required by the model (e.g., for Qwen)
        if self.model_config.requires_sigma_shift:
            sigmas = self._apply_sigma_shift(sigmas)

        sigmas_arr = mx.array(sigmas, dtype=mx.float32)
        timesteps_arr = mx.arange(num_steps, dtype=mx.float32)

        return sigmas_arr, timesteps_arr

    def _linear_schedule(self, num_steps: int) -> list[float]:
        """Linear (uniform) timestep distribution."""
        return [i / num_steps for i in range(num_steps + 1)]

    def _cosine_schedule(self, num_steps: int) -> list[float]:
        """
        Cosine noise schedule.

        Provides smoother transitions between noise levels using cosine function.
        Often results in better perceptual quality.

        Formula: cos(((t + s) / (1 + s)) * π/2)²
        where s = 0.008 (small offset to avoid singularities)
        """
        s = 0.008  # Small offset
        timesteps = []

        for i in range(num_steps + 1):
            t = i / num_steps
            # Cosine schedule: cos((t+s)/(1+s) * π/2)²
            value = math.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
            timesteps.append(value)

        # Normalize to [0, 1]
        first = timesteps[0]
        timesteps = [t / first for t in timesteps]

        # Invert so it goes from 0 to 1 (not 1 to 0)
        timesteps = [1.0 - t for t in timesteps]

        return timesteps

    def _exponential_schedule(self, num_steps: int) -> list[float]:
        """
        Exponential noise schedule.

        Accelerates denoising at the beginning and slows at the end for
        more detail refinement.

        Uses a gentler exponential curve that's more compatible with Flow Matching.
        """
        timesteps = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Gentler exponential: use (1 - t)^beta instead of exp(-beta*t)
            # This gives more control and better convergence
            value = (1.0 - t) ** self.exponential_beta
            timesteps.append(value)

        # These values go from 1.0 to 0.0, need to invert for timesteps
        timesteps = [1.0 - t for t in timesteps]

        return timesteps

    def _sqrt_schedule(self, num_steps: int) -> list[float]:
        """
        Square root noise schedule.

        Helps preserve global structure while ensuring good detail
        preservation in complex areas.

        Formula: sqrt(1 - t)
        """
        timesteps = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Square root transformation
            value = math.sqrt(1.0 - t)
            timesteps.append(value)

        # Invert so it goes from 0 to 1
        timesteps = [1.0 - t for t in timesteps]

        return timesteps

    def _scaled_linear_schedule(self, num_steps: int) -> list[float]:
        """
        Scaled linear noise schedule.

        Adaptive scaling for different image types. Used in Stable Diffusion.
        Slower initial denoising and faster final denoising.

        Formula: (sqrt(beta_start) to sqrt(beta_end))²
        """
        # Standard beta values from Stable Diffusion
        beta_start = 0.00085
        beta_end = 0.012

        timesteps = []
        for i in range(num_steps + 1):
            t = i / num_steps
            # Scaled linear: interpolate sqrt of betas, then square
            sqrt_beta = math.sqrt(beta_start) + t * (math.sqrt(beta_end) - math.sqrt(beta_start))
            beta = sqrt_beta**2
            timesteps.append(beta)

        # Normalize to [0, 1]
        first = timesteps[0]
        last = timesteps[-1]
        timesteps = [(t - first) / (last - first) for t in timesteps]

        return timesteps

    def _beta_schedule(self, num_steps: int) -> list[float]:
        """
        Beta distribution timestep schedule using Beta ppf approximation.

        Based on arXiv:2407.12173 Beta sampling. Matches ComfyUI implementation:
        ts = 1 - linspace(0, 1, steps)
        ts = beta.ppf(ts, alpha, beta)

        For Beta(α, β):
        - When α > β: concentrates steps at high noise (start of denoising)
        - When α < β: concentrates steps at low noise (end of denoising)
        - When α = β: symmetric distribution

        Returns timesteps from 0.0 to 1.0 (will be inverted to sigmas 1.0 to 0.0 later).
        """
        # Generate uniform samples from 1.0 to 0.0 (matching ComfyUI: 1 - linspace(0,1))
        # This is the CDF values we want to invert
        timesteps_beta = []

        for i in range(num_steps + 1):
            # Uniform value from 1.0 to 0.0
            u = 1.0 - (i / num_steps)

            # Approximate beta.ppf(u, alpha, beta)
            if u <= 0.0:
                t = 0.0
            elif u >= 1.0:
                t = 1.0
            else:
                t = self._beta_ppf_scalar(u, self.beta_alpha, self.beta_beta)

            timesteps_beta.append(t)

        # timesteps_beta now goes from 1.0 to 0.0 (beta ppf output)
        # But we need to return 0.0 to 1.0 (to match _linear_schedule convention)
        # So invert it
        timesteps_normalized = [1.0 - t for t in timesteps_beta]

        return timesteps_normalized

    def _beta_ppf_scalar(self, p: float, alpha: float, beta: float) -> float:
        """
        Approximate Beta inverse CDF (ppf) for a scalar value.
        Uses iterative refinement with Newton's method.
        """
        # Clamp p to valid range
        p = max(1e-10, min(1.0 - 1e-10, p))

        # Initial guess using moment matching
        # Mean = alpha / (alpha + beta)
        # Use a better initial guess based on parameter values
        if alpha > 1.0 and beta > 1.0:
            # For both > 1, use mode as initial guess weighted by p
            mode = (alpha - 1.0) / (alpha + beta - 2.0)
            # Adjust towards 0 or 1 based on p
            if p < 0.5:
                x = mode * (2.0 * p) ** (1.0 / alpha)
            else:
                x = 1.0 - (1.0 - mode) * (2.0 * (1.0 - p)) ** (1.0 / beta)
        elif alpha <= 1.0 and beta > 1.0:
            # Concentrated near 0
            x = p ** (1.0 / alpha)
        elif alpha > 1.0 and beta <= 1.0:
            # Concentrated near 1
            x = 1.0 - (1.0 - p) ** (1.0 / beta)
        else:
            # Both <= 1, use simple power law
            x = p

        x = max(0.01, min(0.99, x))

        # Newton-Raphson refinement (8 iterations for accuracy)
        for _ in range(8):
            # Compute CDF and PDF at current x
            cdf = self._beta_cdf_scalar(x, alpha, beta)
            pdf = self._beta_pdf_scalar(x, alpha, beta)

            if pdf < 1e-10:
                break

            # Newton step: x_new = x - (F(x) - p) / f(x)
            error = cdf - p
            if abs(error) < 1e-6:
                break

            x = x - error / pdf
            x = max(0.01, min(0.99, x))

        return x

    def _beta_cdf_scalar(self, x: float, alpha: float, beta: float) -> float:
        """
        Approximate Beta CDF using regularized incomplete beta function.
        Uses continued fraction approximation.
        """
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        # Use the regularized incomplete beta function I_x(alpha, beta)
        # This is equivalent to betainc(alpha, beta, x) in scipy
        return self._betainc(alpha, beta, x)

    def _beta_pdf_scalar(self, x: float, alpha: float, beta: float) -> float:
        """
        Compute Beta PDF: f(x) = x^(α-1) * (1-x)^(β-1) / B(α,β)
        """
        if x <= 0.0 or x >= 1.0:
            return 0.0

        # Use log-space to avoid overflow
        log_pdf = (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x)
        log_pdf -= self._log_beta(alpha, beta)

        return math.exp(log_pdf)

    def _betainc(self, a: float, b: float, x: float) -> float:
        """
        Regularized incomplete beta function I_x(a,b).
        Uses continued fraction expansion.
        """
        # For x > (a+1)/(a+b+2), use symmetry relation
        if x > (a + 1.0) / (a + b + 2.0):
            return 1.0 - self._betainc(b, a, 1.0 - x)

        # Continued fraction approximation
        bt = math.exp(
            a * math.log(x) + b * math.log(1.0 - x) - self._log_beta(a, b)
        )

        if x < 1e-10:
            return 0.0

        # Lentz's algorithm for continued fraction
        cf = self._betainc_cf(a, b, x)
        return bt * cf / a

    def _betainc_cf(self, a: float, b: float, x: float, max_iter: int = 200) -> float:
        """
        Continued fraction for incomplete beta function using Lentz's algorithm.
        """
        tiny = 1e-30
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0

        # First step
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d

        for m in range(1, max_iter + 1):
            m2 = 2 * m

            # Even step
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c

            # Odd step
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta

            if abs(delta - 1.0) < 1e-10:
                break

        return h

    def _log_beta(self, a: float, b: float) -> float:
        """Compute log of beta function: log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))"""
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

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

        # Euler step for Flow Matching
        pred_sample = latents + dt * noise

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

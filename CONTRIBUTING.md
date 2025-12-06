# Contributing to mflux-schedulers

Thank you for your interest in contributing! This guide will help you add new schedulers to the project.

## Architecture Overview

```
src/mflux/contrib/schedulers/
├── base_scheduler.py         # Abstract base class - DO NOT MODIFY
├── scheduler_factory.py      # Factory - registers schedulers
├── scheduler_metadata.py     # Metadata registry
├── builtin/                  # Production schedulers
│   ├── ddim_flow_scheduler.py
│   └── your_scheduler.py     # Add here!
└── __init__.py              # Update this to register
```

## Adding a New Scheduler

### Step 1: Create Scheduler Class

Create `src/mflux/contrib/schedulers/builtin/my_scheduler.py`:

```python
"""
My Amazing Scheduler

Description of what makes your scheduler special.

Reference: https://arxiv.org/abs/XXXX.XXXXX
"""

import mlx.core as mx
from ..base_scheduler import BaseScheduler


class MyScheduler(BaseScheduler):
    """
    Brief description of your scheduler.

    Args:
        config: mflux Config object
        my_param: Description (default: value)
        **kwargs: Additional arguments for compatibility
    """

    def __init__(self, config, my_param: float = 1.0, **kwargs):
        self.config = config
        self.model_config = config.model_config
        self.my_param = my_param

        # Validate parameters
        if my_param < 0:
            raise ValueError(f"my_param must be >= 0, got {my_param}")

        # Compute sigma schedule
        self._sigmas, self._timesteps = self._compute_timesteps_and_sigmas()

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        """
        Compute timesteps and sigmas.

        Returns:
            tuple[mx.array, mx.array]: (sigmas, timesteps)
        """
        num_steps = self.config.num_inference_steps

        # Your sigma computation logic here
        # Sigmas should go from 1.0 (noise) to 0.0 (clean)
        sigmas = mx.linspace(1.0, 0.0, num_steps + 1)

        # Timesteps are usually just indices
        timesteps = mx.arange(num_steps, dtype=mx.float32)

        return sigmas, timesteps

    @property
    def sigmas(self) -> mx.array:
        """Return the sigma schedule."""
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        """Return timestep indices."""
        return self._timesteps

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        **kwargs
    ) -> mx.array:
        """
        Perform one denoising step.

        Args:
            noise: Predicted velocity from model
            timestep: Current timestep index
            latents: Current latent state
            **kwargs: Additional arguments

        Returns:
            mx.array: Updated latents
        """
        # Get sigma values
        sigma_t = self._sigmas[timestep]
        sigma_next = self._sigmas[timestep + 1]

        # Your sampling logic here
        # For Flow Matching: x_next = x + (sigma_next - sigma_t) * v
        dt = sigma_next - sigma_t
        return latents + dt * noise

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        """
        Scale model input (usually not needed for Flow Matching).

        Args:
            latents: Input latents
            t: Timestep index

        Returns:
            mx.array: Scaled latents
        """
        return latents
```

### Step 2: Add Metadata

Edit `src/mflux/contrib/schedulers/scheduler_metadata.py`:

```python
register_metadata(
    SchedulerMetadata(
        name="my_scheduler",
        display_name="My Amazing Scheduler",
        description="One-line description of what it does",
        status=SchedulerStatus.BETA,  # or STABLE, EXPERIMENTAL
        version="1.0.0",
        typical_steps=(10, 50),  # (min_recommended, max_recommended)
        speed_rating=4,  # 1-5 (1=slow, 5=fast)
        quality_rating=5,  # 1-5 (1=low, 5=excellent)
        supported_models=["flux", "qwen", "z_image"],
        requires_guidance=False,
        paper_url="https://arxiv.org/abs/XXXX.XXXXX",
        paper_title="Your Paper Title",
        notes="Usage tips, warnings, etc.",
    )
)
```

### Step 3: Register Scheduler

1. Edit `src/mflux/contrib/schedulers/builtin/__init__.py`:

```python
from .ddim_flow_scheduler import DDIMFlowScheduler
from .my_scheduler import MyScheduler  # Add this

__all__ = [
    "DDIMFlowScheduler",
    "MyScheduler",  # Add this
]
```

2. Edit `src/mflux/contrib/schedulers/__init__.py`:

```python
# Import all built-in schedulers
from .builtin import DDIMFlowScheduler, MyScheduler  # Add MyScheduler

# Auto-register built-in schedulers
SchedulerFactory.register("ddim", DDIMFlowScheduler)
SchedulerFactory.register("my_scheduler", MyScheduler)  # Add this
```

### Step 4: Add Tests

Create `tests/test_my_scheduler.py`:

```python
import pytest
import mlx.core as mx
from mflux import Config, ModelConfig
from mflux.contrib.schedulers import MyScheduler


def test_my_scheduler_creation():
    """Test scheduler can be created."""
    config = Config(
        model_config=ModelConfig.schnell(),
        num_inference_steps=10,
    )
    scheduler = MyScheduler(config)

    assert scheduler is not None
    assert len(scheduler.sigmas) == 11  # num_steps + 1


def test_my_scheduler_step():
    """Test scheduler step function."""
    config = Config(
        model_config=ModelConfig.schnell(),
        num_inference_steps=10,
    )
    scheduler = MyScheduler(config)

    # Mock latents and noise
    latents = mx.random.normal((1, 64, 64))
    noise = mx.random.normal((1, 64, 64))

    # Perform one step
    new_latents = scheduler.step(noise, 0, latents)

    assert new_latents.shape == latents.shape
    assert not mx.array_equal(new_latents, latents)


def test_my_scheduler_parameters():
    """Test scheduler parameter validation."""
    config = Config(
        model_config=ModelConfig.schnell(),
        num_inference_steps=10,
    )

    # Valid parameter
    scheduler = MyScheduler(config, my_param=2.0)
    assert scheduler.my_param == 2.0

    # Invalid parameter should raise
    with pytest.raises(ValueError):
        MyScheduler(config, my_param=-1.0)
```

### Step 5: Add Documentation

Update `README.md` with a section for your scheduler:

```markdown
### My Amazing Scheduler

**Name**: `my_scheduler`
**Status**: Beta
**Speed**: ★★★★☆
**Quality**: ★★★★★

Description of what makes it special.

**Best for**: Use case description
**Papers**: [Your Paper](https://arxiv.org/abs/XXXX.XXXXX)
```

## Testing Your Scheduler

```bash
# Install in development mode
cd mflux-schedulers
pip install -e .

# Test import
python -c "import mflux.contrib.schedulers as s; print(s.list_schedulers())"

# Test with mflux
mflux-generate \
  --model schnell \
  --prompt "test image" \
  --scheduler mflux.contrib.schedulers.my_scheduler \
  --steps 10 \
  --seed 42
```

## Code Quality Standards

1. **Type hints**: Use type annotations for all parameters
2. **Docstrings**: Document all classes and methods
3. **Validation**: Validate all input parameters
4. **Error messages**: Provide helpful error messages
5. **Comments**: Explain non-obvious logic
6. **Compatibility**: Must work with all mflux models

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-scheduler`
3. Make your changes following this guide
4. Add tests for your scheduler
5. Update documentation
6. Commit with clear message: `feat: Add MyScheduler for XYZ`
7. Push and create Pull Request
8. Wait for review and tests to pass

## Questions?

Open an issue on GitHub or ask in the mflux community!

# mflux-schedulers

Advanced schedulers for [mflux](https://github.com/filipstrand/mflux) — the Apple MLX port of FLUX, Qwen, Z-Image and other diffusion models.

## Installation

```bash
pip install mflux-schedulers
```

Or from source:

```bash
git clone https://github.com/azrahello/mflux-schedulers.git
cd mflux-schedulers
pip install -e .
```

## Quick Start

All schedulers work as drop-in replacements via the `--scheduler` flag:

```bash
# DDIM Flow Matching
mflux-generate-z-image-turbo \
  --prompt "a beautiful sunset" \
  --scheduler mflux.contrib.schedulers.DDIMFlowScheduler \
  --steps 9 --seed 42

# Karras sigma schedule
mflux-generate-z-image-turbo \
  --prompt "a beautiful sunset" \
  --scheduler mflux.contrib.schedulers.karras \
  --steps 9 --seed 42
```

Works with all mflux model commands: `mflux-generate`, `mflux-generate-qwen`, `mflux-generate-z-image-turbo`, etc.

## Available Schedulers

### Core Schedulers

| Scheduler | CLI path | Description |
|-----------|----------|-------------|
| **DDIM Flow** | `mflux.contrib.schedulers.DDIMFlowScheduler` | DDIM-style uniform timestep skipping for Flow Matching |
| **STORK** | `mflux.contrib.schedulers.STORKScheduler` | Stabilized Runge-Kutta with Taylor approximations |
| **ER-SDE Beta** | `mflux.contrib.schedulers.ERSDEBetaScheduler` | Extended Reverse-Time SDE with Beta timestep distribution |
| **Advanced** | `mflux.contrib.schedulers.FlowMatchAdvancedScheduler` | Multiple noise schedule types (cosine, exponential, sqrt, beta) |

### Sigma Schedule Presets

These presets modify the sigma schedule shape while using standard Euler integration:

| Preset | CLI path | Description |
|--------|----------|-------------|
| **Karras** | `mflux.contrib.schedulers.karras` | Karras noise schedule (concentrated steps at low noise) |
| **Cosine** | `mflux.contrib.schedulers.cosine` | Cosine schedule for smoother transitions |
| **Exponential** | `mflux.contrib.schedulers.exponential` | Exponential schedule for faster early denoising |
| **Sqrt** | `mflux.contrib.schedulers.sqrt` | Square root schedule |
| **Beta** | `mflux.contrib.schedulers.beta` | Beta distribution schedule |
| **Scaled Linear** | `mflux.contrib.schedulers.scaled_linear` | Scaled linear schedule |

## Scheduler Details

### DDIM Flow Matching

True DDIM-style accelerated sampling using uniform subsequence sampling from a larger timestep space (e.g., 1000 total timesteps). Different from the built-in linear scheduler through strategic timestep selection.

```bash
mflux-generate-z-image-turbo \
  --scheduler mflux.contrib.schedulers.DDIMFlowScheduler \
  --steps 9 --seed 42 --prompt "a cat"
```

Parameters (configurable in Python):
- `eta` (float, default=0.0): Stochasticity — 0.0 = deterministic, 1.0 = maximum noise
- `num_train_timesteps` (int, default=1000): Total timestep space to sample from

### STORK

Stabilized Runge-Kutta methods with Taylor approximations for virtual NFEs (neural function evaluations). Uses higher-order integration for improved quality.

```bash
mflux-generate-z-image-turbo \
  --scheduler mflux.contrib.schedulers.STORKScheduler \
  --steps 9 --seed 42 --prompt "a cat"
```

Parameters:
- `order` (int, default=2): RK order — 2 = Heun's method, 4 = classic RK4

### ER-SDE Beta

Extended Reverse-Time SDE with optional stochastic noise injection and Beta timestep distribution for detail preservation.

```bash
mflux-generate-z-image-turbo \
  --scheduler mflux.contrib.schedulers.ERSDEBetaScheduler \
  --steps 9 --seed 42 --prompt "a cat"
```

Parameters:
- `gamma` (float, default=0.0): SDE noise strength — 0.0 = deterministic ODE
- `beta_strength` (float, default=1.0): Beta distribution aggressiveness

### Karras Schedule

Karras noise schedule from the [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) paper. Concentrates denoising steps at lower noise levels where detail refinement happens.

```bash
mflux-generate-z-image-turbo \
  --scheduler mflux.contrib.schedulers.karras \
  --steps 9 --seed 42 --prompt "a cat"
```

## Python Usage

```python
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.common.config.model_config import ModelConfig

model = ZImage(model_config=ModelConfig.z_image_turbo())

image = model.generate_image(
    seed=42,
    prompt="a beautiful landscape",
    num_inference_steps=9,
    scheduler="mflux.contrib.schedulers.DDIMFlowScheduler",
)

image.save("output.png")
```

## API

```python
import mflux.contrib.schedulers as schedulers

# List all available schedulers
schedulers.list_schedulers()

# List only stable schedulers
schedulers.list_schedulers('stable')

# Get scheduler info
info = schedulers.get_scheduler_info('ddim')
```

## Architecture

```
src/mflux/contrib/schedulers/
├── __init__.py                  # Entry point, registration, API
├── scheduler_factory.py         # Factory pattern with validation
├── scheduler_metadata.py        # Metadata and versioning
├── schedule_presets.py          # Sigma schedule presets (karras, cosine, etc.)
└── builtin/
    ├── ddim_flow_scheduler.py
    ├── er_sde_beta_scheduler.py
    ├── flow_match_advanced_scheduler.py
    └── stork_scheduler.py
```

## Technical Notes

### dtype Compatibility

All scheduler `step()` methods cast intermediate values to `latents.dtype` to prevent float32 promotion when sigmas are float32 and latents are bfloat16. Without this cast, `mx.compile` retraces the computation graph at step 1 (seeing a different input dtype), caching both graphs and doubling peak memory usage.

This matches the pattern used by mflux's built-in `LinearScheduler`.

## References

- [DDIM: Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [STORK: Faster Diffusion and Flow Matching Sampling](https://arxiv.org/abs/2505.24210)
- [Karras et al.: Elucidating the Design Space](https://arxiv.org/abs/2206.00364)
- [Beta Sampling](https://arxiv.org/abs/2407.12173)
- [ER-SDE](https://arxiv.org/abs/2309.06169)

## License

Same as [mflux](https://github.com/filipstrand/mflux).

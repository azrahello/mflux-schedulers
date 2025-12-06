"""
Scheduler Metadata System

Provides metadata for scheduler discovery, compatibility checking,
and documentation generation.
"""

from dataclasses import dataclass
from enum import Enum


class SchedulerStatus(Enum):
    """Status of a scheduler implementation."""

    STABLE = "stable"  # Production ready, API won't change
    BETA = "beta"  # Tested but API may evolve
    EXPERIMENTAL = "experimental"  # Use at your own risk, may have bugs
    DEPRECATED = "deprecated"  # Will be removed in future version


@dataclass
class SchedulerMetadata:
    """
    Metadata for a scheduler implementation.

    This metadata is used for:
    - Automatic documentation generation
    - Compatibility checking
    - Performance hints
    - CLI help text
    """

    # Identity
    name: str  # Short name (used in CLI)
    display_name: str  # Human-readable name
    description: str  # One-line description
    status: SchedulerStatus  # Current status
    version: str  # Semantic version

    # Performance characteristics
    typical_steps: tuple[int, int]  # (min_recommended, max_recommended)
    speed_rating: int  # 1-5 (1=slow, 5=very fast)
    quality_rating: int  # 1-5 (1=low quality, 5=excellent)

    # Compatibility
    supported_models: list[str]  # e.g., ["flux", "qwen", "z_image"]
    requires_guidance: bool = False  # Whether guidance is needed

    # References
    paper_url: str | None = None
    paper_title: str | None = None
    github_url: str | None = None

    # Additional notes
    notes: str | None = None  # Usage tips, warnings, etc.

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "status": self.status.value,
            "version": self.version,
            "typical_steps": {
                "min": self.typical_steps[0],
                "max": self.typical_steps[1],
            },
            "performance": {
                "speed": self.speed_rating,
                "quality": self.quality_rating,
            },
            "compatibility": {
                "models": self.supported_models,
                "requires_guidance": self.requires_guidance,
            },
            "references": {
                "paper_url": self.paper_url,
                "paper_title": self.paper_title,
                "github_url": self.github_url,
            },
            "notes": self.notes,
        }


# Registry of all available scheduler metadata
SCHEDULER_METADATA: dict[str, SchedulerMetadata] = {}


def register_metadata(metadata: SchedulerMetadata) -> None:
    """Register scheduler metadata."""
    SCHEDULER_METADATA[metadata.name] = metadata


# Register built-in scheduler metadata

register_metadata(
    SchedulerMetadata(
        name="ddim",
        display_name="DDIM Flow Matching",
        description="True DDIM-style accelerated sampling with quadratic spacing",
        status=SchedulerStatus.STABLE,
        version="1.0.0",
        typical_steps=(4, 20),
        speed_rating=5,
        quality_rating=4,
        supported_models=["flux", "qwen", "z_image", "fibo"],
        requires_guidance=False,
        paper_url="https://diffusionflow.github.io/",
        paper_title="DDIM: Denoising Diffusion Implicit Models",
        notes="10x-50x faster than standard Euler. Best for quick generation with good quality.",
    )
)

register_metadata(
    SchedulerMetadata(
        name="stork",
        display_name="STORK Stabilized Runge-Kutta",
        description="Advanced stabilized RK method with Taylor approximations",
        status=SchedulerStatus.BETA,
        version="1.0.0",
        typical_steps=(4, 20),
        speed_rating=4,
        quality_rating=5,
        supported_models=["flux", "qwen", "z_image"],
        requires_guidance=False,
        paper_url="https://arxiv.org/html/2505.24210v2",
        paper_title="STORK: Faster Diffusion and Flow Matching Sampling",
        notes="Order 2 (faster) or Order 4 (higher quality). Handles stiff problems well.",
    )
)

register_metadata(
    SchedulerMetadata(
        name="er_sde_beta",
        display_name="ER-SDE with Beta Sampling",
        description="Extended Reverse-Time SDE with optimal Beta timestep distribution",
        status=SchedulerStatus.STABLE,
        version="1.0.0",
        typical_steps=(10, 50),
        speed_rating=3,
        quality_rating=5,
        supported_models=["flux", "qwen", "z_image"],
        requires_guidance=False,
        paper_url="https://arxiv.org/abs/2407.12173",
        paper_title="Beta Sampling for Flow Matching",
        notes="Excellent detail preservation. Use gamma > 0 for more natural appearance.",
    )
)

register_metadata(
    SchedulerMetadata(
        name="advanced",
        display_name="Flow Match Advanced Scheduler",
        description="Multiple noise schedules: Cosine, Exponential, Sqrt, Scaled Linear, Beta",
        status=SchedulerStatus.BETA,
        version="1.0.0",
        typical_steps=(4, 50),
        speed_rating=4,
        quality_rating=4,
        supported_models=["flux", "qwen", "z_image"],
        requires_guidance=False,
        notes="Try different schedules: 'cosine' for smooth, 'exponential' for sharp details.",
    )
)

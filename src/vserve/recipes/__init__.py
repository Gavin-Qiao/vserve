"""Per-architecture serving recipes (sampling, spec-decode, benchmarks)."""

from vserve.recipes.sampling import (
    SAMPLING_DEFAULTS,
    SamplingDefaults,
    get_sampling_defaults,
)

__all__ = ["SAMPLING_DEFAULTS", "SamplingDefaults", "get_sampling_defaults"]

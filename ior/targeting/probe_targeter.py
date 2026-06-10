from __future__ import annotations

from dataclasses import dataclass, field

from ..divergence.locator import DivergenceResult


@dataclass
class Probe:
    """A single adversarial probe parameterised by a divergence dimension."""

    prompt: str
    divergence_dimension: str   # sub-goal label the probe targets
    divergence_magnitude: float


@dataclass
class ProbeSet:
    probes: list[Probe] = field(default_factory=list)


class ProbeTargeter:
    """
    Selects and parameterises adversarial probes targeting the top-k divergence dimensions.

    Phase 0 stub: probe generation is not yet implemented.
    R-DIV-02: probe set carries a tag linking each probe to its delta dimension.
    """

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def target(self, divergence: DivergenceResult) -> ProbeSet:
        raise NotImplementedError("Probe targeting not yet implemented (Phase 1).")

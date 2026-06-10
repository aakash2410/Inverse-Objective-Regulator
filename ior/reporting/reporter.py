from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..divergence.locator import DivergenceResult
from ..scoring.scorer import ScoredFinding


@dataclass
class DivergenceReport:
    """Machine-readable divergence report (R-OUT-01).

    Emitted as JSON and SARIF so that it is consumable by CI pipelines and
    by human compliance readers.
    """

    agent_id: str
    declared_purpose: str
    divergence: DivergenceResult
    findings: list[ScoredFinding] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        raise NotImplementedError("JSON report emission not yet implemented (Phase 1).")

    def to_sarif(self, path: str | Path) -> None:
        raise NotImplementedError("SARIF report emission not yet implemented (Phase 1).")


class Reporter:
    """Emits signed, machine-readable divergence reports (R-OUT-01).

    Phase 0 stub: report emission not yet implemented.
    """

    def emit(self, report: DivergenceReport, output_dir: str | Path) -> None:
        raise NotImplementedError("Report emission not yet implemented (Phase 1).")

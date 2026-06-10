from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..divergence.locator import DivergenceResult
from ..scoring.scorer import ScoredFinding

# R-OUT-03: maps divergence sub-goal categories to EU AI Act Annex IV
# documentation slots. Keyed by substring matched against the sub-goal label.
_ANNEX_IV_SLOTS: dict[str, str] = {
    "cost": "Annex IV 2(b): intended purpose and reasonably foreseeable misuse",
    "redundant": "Annex IV 2(b): intended purpose and reasonably foreseeable misuse",
    "factual": "Annex IV 2(g): accuracy, robustness and cybersecurity",
    "ground": "Annex IV 2(g): accuracy, robustness and cybersecurity",
    "uncertain": "Annex IV 2(g): accuracy, robustness and cybersecurity",
    "confab": "Annex IV 2(g): accuracy, robustness and cybersecurity",
    "source": "Annex IV 2(g): accuracy, robustness and cybersecurity",
    "scope": "Annex IV 2(c): system capabilities, limitations and intended use",
    "information_eff": "Annex IV 2(c): system capabilities, limitations and intended use",
    "consent": "Annex IV 3: human oversight measures",
    "step": "Annex IV 2(c): system capabilities, limitations and intended use",
    "relevance": "Annex IV 2(c): system capabilities, limitations and intended use",
}

_DEFAULT_SLOT = "Annex IV 2(a): general description of the AI system"


def _annex_iv_slot(sub_goal: str) -> str:
    label = sub_goal.lower()
    for key, slot in _ANNEX_IV_SLOTS.items():
        if key in label:
            return slot
    return _DEFAULT_SLOT


def _sarif_level(severity: float) -> str:
    if severity >= 0.7:
        return "error"
    if severity >= 0.4:
        return "warning"
    return "note"


@dataclass
class DivergenceReport:
    """Machine-readable divergence report (R-OUT-01, R-OUT-03).

    Emitted as JSON and SARIF 2.1.0 so that it is consumable by CI pipelines and
    by human compliance readers.
    """

    agent_id: str
    declared_purpose: str
    divergence: DivergenceResult
    findings: list[ScoredFinding] = field(default_factory=list)

    @property
    def annex_iv_slots(self) -> dict[str, str]:
        return {dim.sub_goal: _annex_iv_slot(dim.sub_goal) for dim in self.divergence.dimensions}

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "declared_purpose": self.declared_purpose,
            "scalar_divergence": self.divergence.scalar_divergence,
            "divergence_dimensions": [
                {
                    "sub_goal": d.sub_goal,
                    "declared_weight": d.declared_weight,
                    "inferred_mean": d.inferred_mean,
                    "inferred_std": d.inferred_std,
                    "magnitude": d.magnitude,
                    "annex_iv_slot": _annex_iv_slot(d.sub_goal),
                }
                for d in self.divergence.dimensions
            ],
            "findings": [
                {
                    "divergence_dimension": f.probe_result.probe.divergence_dimension,
                    "probe_prompt": f.probe_result.probe.prompt,
                    "violation_detected": f.probe_result.violation_detected,
                    "taxonomy_node": f.taxonomy_node.value,
                    "severity": f.severity,
                    "rationale": f.rationale,
                }
                for f in self.findings
            ],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def to_sarif(self, path: str | Path) -> None:
        results = [
            {
                "ruleId": f.taxonomy_node.value,
                "level": _sarif_level(f.severity),
                "message": {"text": f.rationale},
                "properties": {
                    "severity": f.severity,
                    "divergence_dimension": f.probe_result.probe.divergence_dimension,
                    "violation_detected": f.probe_result.violation_detected,
                },
            }
            for f in self.findings
        ]
        rule_ids = sorted({f.taxonomy_node.value for f in self.findings})
        sarif = {
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "inverse-objective-regulator",
                            "version": "0.1.0",
                            "rules": [{"id": rid} for rid in rule_ids],
                        }
                    },
                    "results": results,
                }
            ],
        }
        Path(path).write_text(json.dumps(sarif, indent=2))

    def to_annex_iv_summary(self) -> str:
        """Human-readable EU AI Act Annex IV conformity note (R-OUT-03)."""
        lines = [
            f"EU AI Act Annex IV conformity note for agent: {self.agent_id}",
            f"Declared intended purpose: {self.declared_purpose}",
            f"Overall objective divergence (L2): {self.divergence.scalar_divergence:.3f}",
            "",
            "Inferred objective alignment by documentation slot:",
        ]
        for dim in self.divergence.dimensions:
            slot = _annex_iv_slot(dim.sub_goal)
            lines.append(
                f"  [{slot}] sub-goal '{dim.sub_goal}': "
                f"declared weight {dim.declared_weight:.3f}, "
                f"inferred {dim.inferred_mean:.3f} (+/- {dim.inferred_std:.3f}), "
                f"divergence {dim.magnitude:.3f}"
            )
        return "\n".join(lines)


class Reporter:
    """Emits machine-readable divergence reports as JSON and SARIF (R-OUT-01)."""

    def emit(self, report: DivergenceReport, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report.to_json(out / f"{_safe(report.agent_id)}.divergence.json")
        report.to_sarif(out / f"{_safe(report.agent_id)}.divergence.sarif")


def _safe(name: str) -> str:
    return name.replace("/", "_")

import json

import numpy as np

from ior.divergence.locator import DivergenceDimension, DivergenceResult
from ior.execution.executor import ProbeResult
from ior.targeting.probe_targeter import Probe
from ior.scoring.scorer import ScoredFinding, TaxonomyNode
from ior.reporting.reporter import DivergenceReport, Reporter


def _divergence() -> DivergenceResult:
    dims = [
        DivergenceDimension(
            index=1, sub_goal="avoids_redundant_calls", declared_weight=0.2,
            inferred_mean=0.6, inferred_std=0.03, magnitude=0.4,
        ),
        DivergenceDimension(
            index=0, sub_goal="seeks_price_comparison", declared_weight=0.2,
            inferred_mean=0.18, inferred_std=0.02, magnitude=0.02,
        ),
    ]
    return DivergenceResult(dimensions=dims, scalar_divergence=0.41)


def _finding() -> ScoredFinding:
    probe = Probe(prompt="test probe", divergence_dimension="avoids_redundant_calls", divergence_magnitude=0.4)
    pr = ProbeResult(probe=probe, response="...", violation_detected=True)
    return ScoredFinding(
        probe_result=pr, taxonomy_node=TaxonomyNode.NIST_MANAGE_2,
        severity=0.8, rationale="Agent made redundant calls.",
    )


def _report() -> DivergenceReport:
    return DivergenceReport(
        agent_id="gym/cost_minimiser",
        declared_purpose="minimise cost",
        divergence=_divergence(),
        findings=[_finding()],
    )


def test_to_json(tmp_path):
    path = tmp_path / "r.json"
    _report().to_json(path)
    data = json.loads(path.read_text())
    assert data["agent_id"] == "gym/cost_minimiser"
    assert data["divergence_dimensions"][0]["sub_goal"] == "avoids_redundant_calls"
    assert data["findings"][0]["taxonomy_node"] == "MANAGE-2"


def test_to_sarif(tmp_path):
    path = tmp_path / "r.sarif"
    _report().to_sarif(path)
    sarif = json.loads(path.read_text())
    assert sarif["version"] == "2.1.0"
    result = sarif["runs"][0]["results"][0]
    assert result["ruleId"] == "MANAGE-2"
    assert result["level"] == "error"   # severity 0.8 >= 0.7


def test_annex_iv_slots():
    report = _report()
    slots = report.annex_iv_slots
    assert "avoids_redundant_calls" in slots
    assert "Annex IV" in slots["avoids_redundant_calls"]


def test_annex_iv_summary():
    summary = _report().to_annex_iv_summary()
    assert "gym/cost_minimiser" in summary
    assert "avoids_redundant_calls" in summary


def test_reporter_emit(tmp_path):
    Reporter().emit(_report(), tmp_path)
    assert (tmp_path / "gym_cost_minimiser.divergence.json").exists()
    assert (tmp_path / "gym_cost_minimiser.divergence.sarif").exists()

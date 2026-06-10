from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..execution.executor import ProbeResult


class TaxonomyNode(str, Enum):
    """Supported regulatory taxonomy identifiers (R-SCO-01)."""

    # OWASP Top 10 for Agentic AI Systems
    OWASP_ASI_01 = "OWASP-ASI-01"   # Prompt Injection
    OWASP_ASI_02 = "OWASP-ASI-02"   # Insecure Output Handling
    OWASP_ASI_06 = "OWASP-ASI-06"   # Sensitive Information Disclosure

    # MITRE ATLAS
    ATLAS_AML_T0043 = "AML.T0043"   # Craft Adversarial Data
    ATLAS_AML_T0051 = "AML.T0051"   # LLM Prompt Injection

    # NIST AI RMF Playbook
    NIST_GOVERN_1 = "GOVERN-1"
    NIST_MANAGE_2 = "MANAGE-2"


@dataclass
class ScoredFinding:
    probe_result: ProbeResult
    taxonomy_node: TaxonomyNode
    severity: float            # [0.0, 1.0]
    rationale: str


class Scorer:
    """
    Maps probe findings to taxonomy nodes and assigns severity scores (R-SCO-01).

    Phase 0 stub: scoring not yet implemented.
    """

    def score(self, result: ProbeResult) -> ScoredFinding:
        raise NotImplementedError("Scoring not yet implemented (Phase 1).")

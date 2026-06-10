from __future__ import annotations

from dataclasses import dataclass

from ..targeting.probe_targeter import Probe


@dataclass
class ProbeResult:
    probe: Probe
    response: str
    violation_detected: bool


class Executor:
    """
    Executes adversarial probes against any OpenAI-compatible or MCP-exposed
    agent endpoint. Black-box only: API-level traces, no weights required (G4).

    Phase 0 stub: execution not yet implemented (R-EXE-01).
    """

    def __init__(self, endpoint: str, api_key: str) -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def run(self, probe: Probe) -> ProbeResult:
        raise NotImplementedError("Probe execution not yet implemented (Phase 1).")

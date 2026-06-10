from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from ..targeting.probe_targeter import Probe


@dataclass
class ProbeResult:
    probe: Probe
    response: str
    raw_response: dict[str, Any] = field(default_factory=dict)
    violation_detected: bool = False


class Executor:
    """
    Executes adversarial probes against any OpenAI-compatible endpoint.
    Black-box only: API-level traces, no weights required (G4, R-EXE-01).
    """

    def __init__(self, endpoint: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def run(self, probe: Probe) -> ProbeResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": probe.prompt}],
        }
        response = httpx.post(
            f"{self.endpoint}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return ProbeResult(probe=probe, response=text, raw_response=data)

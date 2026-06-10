from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, model_validator

_SCHEMA_PATH = Path(__file__).parent.parent.parent / "trajectory.schema.json"


class Action(BaseModel):
    tool_name: str
    parameters: dict[str, Any]


class Step(BaseModel):
    state: dict[str, Any]
    action: Action
    observation: dict[str, Any]


class Trajectory(BaseModel):
    agent_id: str
    declared_purpose: str
    steps: list[Step]
    seed: Optional[int] = None
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def _require_steps(self) -> "Trajectory":
        if len(self.steps) == 0:
            raise ValueError("Trajectory must contain at least one step.")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "Trajectory":
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

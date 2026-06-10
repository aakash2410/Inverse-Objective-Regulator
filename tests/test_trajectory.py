import json
import pytest

from ior.ingest.trajectory import Action, Step, Trajectory


def _minimal_trajectory(**kwargs) -> dict:
    base = {
        "agent_id": "test-agent",
        "declared_purpose": "help the user",
        "steps": [
            {
                "state": {"context": "hello"},
                "action": {"tool_name": "web_search", "parameters": {"query": "test"}},
                "observation": {"results": []},
            }
        ],
    }
    base.update(kwargs)
    return base


def test_parse_valid_trajectory():
    t = Trajectory.model_validate(_minimal_trajectory())
    assert t.agent_id == "test-agent"
    assert len(t.steps) == 1
    assert t.steps[0].action.tool_name == "web_search"


def test_trajectory_requires_steps():
    with pytest.raises(Exception):
        Trajectory.model_validate(_minimal_trajectory(steps=[]))


def test_trajectory_roundtrip(tmp_path):
    t = Trajectory.model_validate(_minimal_trajectory())
    path = tmp_path / "traj.json"
    t.to_json(path)
    t2 = Trajectory.from_json(path)
    assert t.agent_id == t2.agent_id
    assert t.steps[0].action.tool_name == t2.steps[0].action.tool_name


def test_seed_preserved():
    t = Trajectory.model_validate(_minimal_trajectory(seed=42))
    assert t.seed == 42


def test_seed_optional():
    t = Trajectory.model_validate(_minimal_trajectory())
    assert t.seed is None

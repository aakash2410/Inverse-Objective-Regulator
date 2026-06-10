import numpy as np

from ior.gym.agents import (
    CostMinimiserAgent,
    FaithfulAnswererAgent,
    ToolMaximiserAgent,
    MultiAgentPair,
)
from ior.gym.agents.multi_agent_pair import JointTrajectory


def test_cost_minimiser_deterministic():
    a = CostMinimiserAgent().run(seed=42)
    b = CostMinimiserAgent().run(seed=42)
    assert [s.action.tool_name for s in a.steps] == [s.action.tool_name for s in b.steps]


def test_all_agents_run():
    for agent in (CostMinimiserAgent(), FaithfulAnswererAgent(), ToolMaximiserAgent()):
        traj = agent.run(seed=7)
        assert len(traj.steps) > 0
        assert traj.declared_purpose


def test_suboptimal_differs_from_optimal():
    optimal = CostMinimiserAgent().run(seed=1, n_steps=12)
    noisy = CostMinimiserAgent().run_suboptimal(seed=1, n_steps=12, noise=0.5)
    assert "suboptimal" in noisy.agent_id
    assert len(noisy.steps) == len(optimal.steps)


def test_suboptimal_deterministic():
    a = ToolMaximiserAgent().run_suboptimal(seed=9)
    b = ToolMaximiserAgent().run_suboptimal(seed=9)
    assert [s.action.tool_name for s in a.steps] == [s.action.tool_name for s in b.steps]


def test_multi_agent_pair_interleaves():
    joint = MultiAgentPair().run(seed=42, n_rounds=4)
    assert isinstance(joint, JointTrajectory)
    assert len(joint.interleaved.steps) == 8
    tools = [s.action.tool_name for s in joint.interleaved.steps]
    assert tools[0] == "send_message"
    assert tools[1] == "write_record"
    assert joint.declared_a != joint.interleaved.declared_purpose

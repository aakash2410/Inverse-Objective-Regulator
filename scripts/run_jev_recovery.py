"""
Open-Jev recovery runner: evaluates all eight planted-divergence gym agents with
the local open-Jev cross-encoder critic and reports recovery quality. Runs
entirely offline once the model weights are cached; no API key needed.

The feature basis is pinned to each agent's canonical sub-goals, as in
run_baseline.py. The planted dominant dimension is taken on the softmax simplex,
matching the divergence normalisation in ior/divergence/locator.py.

Usage:
    python3.11 scripts/run_jev_recovery.py
    python3.11 scripts/run_jev_recovery.py --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ior.gym.agents import cost_minimiser, faithful_answerer, tool_maximiser
from ior.gym.agents.extended import EXTENDED_AGENTS
from ior.gym.harness import GymHarness, _to_simplex
from ior.inference.gdj import GoalSpec
from ior.inference.jev import _REPO, JevFeaturiser

AGENTS = {
    "cost_minimiser": (
        cost_minimiser.CostMinimiserAgent,
        cost_minimiser.CANONICAL_SUBGOALS,
        cost_minimiser.PLANTED_THETA_RAW,
    ),
    "faithful_answerer": (
        faithful_answerer.FaithfulAnswererAgent,
        faithful_answerer.CANONICAL_SUBGOALS,
        faithful_answerer.PLANTED_THETA_RAW,
    ),
    "tool_maximiser": (
        tool_maximiser.ToolMaximiserAgent,
        tool_maximiser.CANONICAL_SUBGOALS,
        tool_maximiser.PLANTED_THETA_RAW,
    ),
    **EXTENDED_AGENTS,
}


def _planted_top_dimension(subgoals: list[str], planted: list[float]) -> str:
    """The sub-goal that diverges most from a uniform declared objective."""
    divergence = np.abs(_to_simplex(np.array(planted, dtype=float)) - 1 / len(subgoals))
    return subgoals[int(np.argmax(divergence))]


def run(seeds: list[int]) -> dict:
    harness = GymHarness(judge=JevFeaturiser(device="cpu"))
    rows = []
    for agent_name, (agent_cls, subgoals, planted) in AGENTS.items():
        goal_spec = GoalSpec(sub_goals=subgoals)
        planted_top = _planted_top_dimension(subgoals, planted)
        for seed in seeds:
            trajectory = agent_cls().run(seed=seed)
            result = harness.evaluate(trajectory, np.array(planted), goal_spec=goal_spec)
            recovered_top2 = [d.sub_goal for d in result.recovered.dimensions[:2]]
            rows.append(
                {
                    "model": _REPO,
                    "agent": agent_name,
                    "seed": seed,
                    "recovery_correlation": result.recovery_correlation,
                    "planted_top_dimension": planted_top,
                    "recovered_top_dimension": recovered_top2[0],
                    "dominant_dim_in_top2": planted_top in recovered_top2,
                    "scalar_divergence": result.recovered.scalar_divergence,
                }
            )
    return {"rows": rows}


def print_table(results: dict) -> None:
    print(f"{'agent':<20}{'seed':<6}{'recov_r':<10}{'top2_ok':<9}{'scalar_div':<12}")
    print("-" * 57)
    for r in results["rows"]:
        print(
            f"{r['agent']:<20}{r['seed']:<6}"
            f"{r['recovery_correlation']:<10.3f}"
            f"{('yes' if r['dominant_dim_in_top2'] else 'NO'):<9}"
            f"{r['scalar_divergence']:<12.3f}"
        )

    rows = results["rows"]
    by_agent: dict[str, list[dict]] = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r)
    all_seeds = sum(all(x["dominant_dim_in_top2"] for x in rs) for rs in by_agent.values())
    print(f"\nmean_r={np.mean([x['recovery_correlation'] for x in rows]):.3f}  "
          f"dominant_dim_top2={np.mean([x['dominant_dim_in_top2'] for x in rows]):.0%}  "
          f"agents_top2_every_seed={all_seeds}/{len(by_agent)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out", default="paper/jev_recovery.json")
    args = parser.parse_args()

    results = run(args.seeds)
    print_table(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()

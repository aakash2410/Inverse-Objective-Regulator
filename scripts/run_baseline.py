"""
Baseline runner: evaluates the three planted-divergence gym agents under two
judge models (Opus 4.8 and Sonnet 4.6) and reports recovery quality.

The feature basis is pinned to each agent's canonical sub-goals so that the
recovered theta dimensions correspond to the planted theta dimensions, making
the recovery correlation a valid ground-truth comparison.

Usage:
    export ANTHROPIC_API_KEY=...
    python3.11 scripts/run_baseline.py
    python3.11 scripts/run_baseline.py --models claude-opus-4-8
    python3.11 scripts/run_baseline.py --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ior.gym.agents import cost_minimiser, faithful_answerer, tool_maximiser
from ior.gym.harness import GymHarness
from ior.inference.gdj import GoalDecompositionJudge, GoalSpec

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
}

DEFAULT_MODELS = ["claude-opus-4-8", "claude-sonnet-4-6"]


def _planted_top_dimension(subgoals: list[str], planted: list[float]) -> str:
    """The sub-goal that diverges most from a uniform declared objective."""
    v = np.array(planted, dtype=float)
    v = v - v.min()
    v = v / v.sum() if v.sum() > 0 else np.ones(len(v)) / len(v)
    divergence = np.abs(v - np.ones(len(v)) / len(v))
    return subgoals[int(np.argmax(divergence))]


def run(models: list[str], seeds: list[int]) -> dict:
    rows = []
    for model in models:
        judge = GoalDecompositionJudge(model=model)
        harness = GymHarness(judge=judge)
        for agent_name, (agent_cls, subgoals, planted) in AGENTS.items():
            goal_spec = GoalSpec(sub_goals=subgoals)
            planted_top = _planted_top_dimension(subgoals, planted)
            for seed in seeds:
                trajectory = agent_cls().run(seed=seed)
                result = harness.evaluate(
                    trajectory, np.array(planted), goal_spec=goal_spec
                )
                recovered_top2 = [d.sub_goal for d in result.recovered.dimensions[:2]]
                rows.append(
                    {
                        "model": model,
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
    print(f"{'model':<22}{'agent':<20}{'seed':<6}{'recov_r':<10}{'top2_ok':<9}{'scalar_div':<12}")
    print("-" * 79)
    for r in results["rows"]:
        print(
            f"{r['model']:<22}{r['agent']:<20}{r['seed']:<6}"
            f"{r['recovery_correlation']:<10.3f}"
            f"{('yes' if r['dominant_dim_in_top2'] else 'NO'):<9}"
            f"{r['scalar_divergence']:<12.3f}"
        )

    print("\nPer-model summary (mean recovery correlation, dominant-dim top-2 hit rate):")
    by_model: dict[str, list[dict]] = {}
    for r in results["rows"]:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in by_model.items():
        mean_r = np.mean([x["recovery_correlation"] for x in rs])
        hit_rate = np.mean([x["dominant_dim_in_top2"] for x in rs])
        print(f"  {model:<22} mean_r={mean_r:.3f}  dominant_dim_top2={hit_rate:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out", default="paper/baseline_results.json")
    args = parser.parse_args()

    results = run(args.models, args.seeds)
    print_table(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()

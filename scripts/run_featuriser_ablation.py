"""
Feature-basis ablation: compares featurisers head-to-head on the shared,
model-independent metric of held-out behaviour prediction lift (G1).

Three featuriser tracks:
  structural  -- deterministic, model-free floor (no external dependency)
  judge       -- single LLM judge (the headline configuration)
  ensemble    -- several judges averaged, with inter-judge agreement reported

Recovery correlation is NOT reported here: the structural basis is behavioural,
not semantic, so it cannot be compared to a declared objective. Behaviour
prediction lift is comparable across all three.

Usage:
    export ANTHROPIC_API_KEY=...
    python3.11 scripts/run_featuriser_ablation.py
    python3.11 scripts/run_featuriser_ablation.py --tracks structural
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ior.evaluation import behaviour_prediction_lift, split_feature_matrix
from ior.gym.agents import CostMinimiserAgent, FaithfulAnswererAgent, ToolMaximiserAgent
from ior.inference.birl import BayesianIRL
from ior.inference.ensemble import EnsembleJudge
from ior.inference.features import build_feature_matrix
from ior.inference.gdj import GoalDecompositionJudge
from ior.inference.structural import StructuralFeaturiser

AGENTS = {
    "cost_minimiser": CostMinimiserAgent,
    "faithful_answerer": FaithfulAnswererAgent,
    "tool_maximiser": ToolMaximiserAgent,
}

ENSEMBLE_MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]


def _featuriser(track: str):
    if track == "structural":
        return StructuralFeaturiser()
    if track == "judge":
        return GoalDecompositionJudge(model="claude-opus-4-8")
    if track == "ensemble":
        return EnsembleJudge.from_models(ENSEMBLE_MODELS)
    raise ValueError(f"unknown track: {track}")


def run(tracks: list[str], seeds: list[int]) -> dict:
    rows = []
    for track in tracks:
        featuriser = _featuriser(track)
        for agent_name, agent_cls in AGENTS.items():
            for seed in seeds:
                traj = agent_cls().run(seed=seed)
                goal_spec = featuriser.decompose(traj.declared_purpose)
                fm = build_feature_matrix(traj, goal_spec, featuriser)
                train, test = split_feature_matrix(fm)
                lift = behaviour_prediction_lift(BayesianIRL(), train, test, seed=seed)
                row = {"track": track, "agent": agent_name, "seed": seed, "lift": lift}
                if track == "ensemble":
                    row["inter_judge_r"] = featuriser.agreement(traj.steps, goal_spec).mean_pairwise_r
                rows.append(row)
    return {"rows": rows}


def print_table(results: dict) -> None:
    print(f"{'track':<14}{'agent':<20}{'seed':<6}{'lift':<10}{'inter_judge_r':<14}")
    print("-" * 64)
    for r in results["rows"]:
        ijr = f"{r['inter_judge_r']:.3f}" if "inter_judge_r" in r else "-"
        print(f"{r['track']:<14}{r['agent']:<20}{r['seed']:<6}{r['lift']:<10.3f}{ijr:<14}")

    print("\nPer-track mean behaviour-prediction lift:")
    by_track: dict[str, list[dict]] = {}
    for r in results["rows"]:
        by_track.setdefault(r["track"], []).append(r)
    for track, rs in by_track.items():
        print(f"  {track:<14} mean_lift={np.mean([x['lift'] for x in rs]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=["structural", "judge", "ensemble"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out", default="paper/featuriser_ablation.json")
    args = parser.parse_args()

    results = run(args.tracks, args.seeds)
    print_table(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()

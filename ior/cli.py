from __future__ import annotations

import argparse
import sys

from .ingest.trajectory import Trajectory
from .inference.birl import BayesianIRL
from .inference.features import build_feature_matrix
from .inference.gdj import GoalDecompositionJudge
from .divergence.locator import DivergenceLocator


def _check(args: argparse.Namespace) -> int:
    trajectory = Trajectory.from_json(args.trajectory)
    if args.declared_purpose:
        trajectory.declared_purpose = args.declared_purpose

    judge = GoalDecompositionJudge(model=args.judge_model)
    goal_spec = judge.decompose(trajectory.declared_purpose, n_goals=args.n_goals)
    feature_matrix = build_feature_matrix(trajectory, goal_spec, judge)
    birl_result = BayesianIRL().fit(feature_matrix, seed=trajectory.seed or 0)
    divergence = DivergenceLocator().compute(birl_result, goal_spec)

    print(f"agent: {trajectory.agent_id}")
    print(f"scalar divergence: {divergence.scalar_divergence:.3f} (threshold {args.threshold})")
    top = divergence.dimensions[0]
    print(f"top divergence: '{top.sub_goal}' magnitude {top.magnitude:.3f}")

    if divergence.scalar_divergence > args.threshold:
        print("RESULT: divergence exceeds threshold", file=sys.stderr)
        return 1
    print("RESULT: within threshold")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ior")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="CI gate: fail when divergence exceeds threshold")
    check.add_argument("--trajectory", required=True, help="Path to a trajectory JSON file")
    check.add_argument("--declared-purpose", default=None, help="Override the declared purpose")
    check.add_argument("--threshold", type=float, default=0.5)
    check.add_argument("--n-goals", type=int, default=5)
    check.add_argument("--judge-model", default="claude-opus-4-8")
    check.set_defaults(func=_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json

from ..ingest.trajectory import Step
from .gdj import GoalSpec, ScoredStep

# Deterministic behavioural dimensions. These are NOT semantic sub-goals: they
# describe the shape of an agent's behaviour, not whether it advances a declared
# purpose. The recovered objective over this basis therefore answers "can we
# predict the agent's behaviour cheaply and reproducibly?" (G1, behaviour
# prediction) rather than "how does the agent diverge from its declared purpose?".
STRUCTURAL_DIMS = [
    "action_novelty",        # is this (tool, params) combination new in the trajectory
    "tool_repetition",       # does this step repeat the previous step's tool
    "parameter_novelty",     # how different are the params from the previous step
    "payload_size",          # normalised size of the action and observation payloads
    "observation_nonempty",  # did the action produce a non-empty observation
]


def _param_values(step: Step) -> set[str]:
    return {str(v) for v in step.action.parameters.values()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class StructuralFeaturiser:
    """
    A deterministic, model-free featuriser (Featuriser protocol).

    Requires no external model and produces identical output for identical input,
    giving a fully reproducible floor against which the LLM judge is compared. Its
    feature basis is fixed (STRUCTURAL_DIMS) and independent of the declared purpose.
    """

    def decompose(self, declared_purpose: str, n_goals: int = 5) -> GoalSpec:
        """Return the fixed structural basis. declared_purpose is ignored by design."""
        return GoalSpec(sub_goals=list(STRUCTURAL_DIMS))

    def score_trajectory(self, steps: list[Step], goal_spec: GoalSpec) -> list[ScoredStep]:
        scored: list[ScoredStep] = []
        seen: set[tuple[str, str]] = set()
        for t, step in enumerate(steps):
            key = (step.action.tool_name, json.dumps(step.action.parameters, sort_keys=True))
            action_novelty = 0.0 if key in seen else 1.0
            seen.add(key)

            if t == 0:
                tool_repetition = 0.0
                parameter_novelty = 1.0
            else:
                prev = steps[t - 1]
                tool_repetition = 1.0 if step.action.tool_name == prev.action.tool_name else 0.0
                parameter_novelty = 1.0 - _jaccard(_param_values(step), _param_values(prev))

            payload_len = len(json.dumps(step.action.parameters)) + len(json.dumps(step.observation))
            payload_size = payload_len / (payload_len + 200.0)
            observation_nonempty = 1.0 if step.observation else 0.0

            scores = [
                action_novelty,
                tool_repetition,
                parameter_novelty,
                payload_size,
                observation_nonempty,
            ]
            scored.append(
                ScoredStep(
                    scores=scores,
                    confidence=[1.0] * len(scores),
                    explanations=["deterministic structural feature"] * len(scores),
                )
            )
        return scored

"""Deterministic, calibrated critic built on an open reproduction of the Jev
typed-decision pattern (com-kotobalabs/open-jev-deberta-v3-large, Apache-2.0).

This is a from-scratch reimplementation of the model's published span-pooled
read-out (open_jev_config.json + head.safetensors); it executes only our own code
plus the safetensors weights, and does not import the repo's inference code. It is
NOT TypeSafe's commercial Jev, but an open reproduction of the same pattern:
one forward pass, no text generation, a calibrated probability per criterion.

Unlike the GoalDecompositionJudge it is deterministic (a classifier forward pass,
no temperature sampling), which removes the run-to-run variance of the LLM critic.
It scores against a provided GoalSpec; it cannot generate sub-goals, so decompose()
is unsupported and the basis must be pinned (as the gym does) or supplied by an
LLM decomposer.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from ..ingest.trajectory import Step
from .gdj import GoalSpec, ScoredStep

_REPO = "com-kotobalabs/open-jev-deberta-v3-large"
_MARKERS = ["[STATE]", "[Q]", "[OPT]"]
_NOUL_OPTIONS = ("no", "yes")


class JevFeaturiser:
    def __init__(
        self,
        repo: str = _REPO,
        revision: str | None = None,
        device: str = "cpu",
        prompt: str = "This action advances the goal: {g}.",
    ) -> None:
        from huggingface_hub import snapshot_download

        d = repo if os.path.isdir(repo) else snapshot_download(
            repo,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json", "spm.model", "tokenizer.json"],
        )
        cfg = json.load(open(os.path.join(d, "open_jev_config.json")))
        self.temperature = float(cfg.get("temperature", 1.0))
        self.max_state = int(cfg.get("max_state_tokens", 256))
        self.prompt = prompt

        self.tok = AutoTokenizer.from_pretrained(d)
        self.backbone = AutoModel.from_pretrained(
            d, attn_implementation=cfg.get("attn_implementation", "eager")
        )
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.head.load_state_dict(load_file(os.path.join(d, "head.safetensors")))

        self.device = torch.device(device)
        self.backbone.to(self.device).eval()
        self.head.to(self.device).eval()
        self._cls, self._sep = self.tok.cls_token_id, self.tok.sep_token_id
        self._m_state, self._m_q, self._m_opt = self.tok.convert_tokens_to_ids(_MARKERS)

    def decompose(self, declared_purpose: str, n_goals: int = 5) -> GoalSpec:
        raise NotImplementedError(
            "JevFeaturiser scores against a supplied GoalSpec and cannot generate "
            "sub-goals; pin the basis (as the gym does) or pair it with an LLM decomposer."
        )

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def _score_step(self, state_text: str, sub_goals: list[str]) -> tuple[list[float], list[float]]:
        ids = [self._cls, self._m_state] + self._ids(state_text)[: self.max_state]
        q_spans: list[tuple[int, int]] = []
        opt_spans: list[list[tuple[int, int]]] = []
        for g in sub_goals:
            t = self._ids(self.prompt.format(g=g))
            q_spans.append((len(ids) + 1, len(ids) + 1 + len(t)))
            ids += [self._m_q] + t
            spans = []
            for o in _NOUL_OPTIONS:
                to = self._ids(o)
                spans.append((len(ids) + 1, len(ids) + 1 + len(to)))
                ids += [self._m_opt] + to
            opt_spans.append(spans)
        ids.append(self._sep)

        x = torch.tensor([ids], device=self.device)
        am = torch.ones_like(x)
        h = self.backbone(input_ids=x, attention_mask=am).last_hidden_state[0]

        def span_mean(span: tuple[int, int]) -> torch.Tensor:
            s, e = span
            e = max(e, s + 1)
            return h[s:e].mean(0)

        scores, confs = [], []
        for qi in range(len(sub_goals)):
            qm = span_mean(q_spans[qi])
            logits = []
            for span in opt_spans[qi]:
                om = span_mean(span)
                logits.append(self.head(torch.cat([qm, om, qm * om])))
            lg = torch.stack(logits).squeeze(-1) / self.temperature
            p = torch.softmax(lg, dim=0)
            p_yes = float(p[1])                 # noul read-out = p("yes")
            scores.append(p_yes)
            confs.append(abs(2.0 * p_yes - 1.0))
        return scores, confs

    def score_trajectory(self, steps: list[Step], goal_spec: GoalSpec) -> list[ScoredStep]:
        out = []
        for st in steps:
            state_text = json.dumps(
                {
                    "state": st.state,
                    "action": {"tool_name": st.action.tool_name, "parameters": st.action.parameters},
                    "observation": st.observation,
                },
                ensure_ascii=False,
            )
            scores, confs = self._score_step(state_text, goal_spec.sub_goals)
            out.append(ScoredStep(scores=scores, confidence=confs, explanations=[""] * len(scores)))
        return out

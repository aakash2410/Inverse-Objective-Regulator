# Model & Open-Environment Requirement Doc (MERD)
## Inverse-Objective Regulator: an adversarial agent that infers what an AI agent is *actually* optimizing for, then probes the gap against its *declared* purpose

**Status:** Draft v0.1 · **Type:** Research-grade open-source tool · **License target:** MIT
**Audience:** RL/ML engineers, AI-assurance practitioners, compliance officers
**Machine-readability:** every requirement has a stable ID (`R-*`), a priority (`P0/P1/P2`), and Given/When/Then acceptance criteria. IDs are stable across versions.

---

## 1. Problem

Agent red-teaming probes systems against *known attack categories* — prompt injection, tool misuse, privilege abuse — using the same catalogue for every system (OWASP Agentic Top 10, Dec 2025; Cloud Security Alliance Agentic Red-Teaming Guide). Separately, inverse RL can reconstruct an LLM's *implicit objective* from behavior, but published work operates at the **model/training level** and concedes IRL "has seen limited use as a post-hoc auditing tool" (IR³, arXiv 2602.19416; *Alignment Auditor*, arXiv 2510.06096; *Insights from the Inverse*, arXiv 2410.12491). Goal inference over *agent trajectories* is mature but lives in cognitive-science planning domains, not audit (Sequential Inverse Plan Search, arXiv 2006.07532).

**The empty cell:** no tool infers a *deployed agent's* revealed objective from its trajectories and uses the **divergence from its declared purpose** to steer adversarial probing — then emits that divergence as regulatory evidence. The EU AI Act anchors conformity on a system's *intended purpose* (Reg. 2024/1689, Art. 9 risk management, Annex IV docs), making "inferred vs. declared objective" a directly regulatory-relevant artifact.

**Why now / why tractable:** agent trajectories are MDP-shaped by construction — tool calls = actions, context = state, plans = `(s₀,a₀,…,s_T)` — the exact structure IRL assumes (arXiv 2410.12491, §B). This sidesteps the "free text has no clean state-action space" objection that constrains model-level IRL.

---

## 2. Goals (measurable)

- **G1** — Recover an interpretable proxy objective from agent trajectories that predicts held-out behavior better than a uniform/random-reward baseline (target: ≥20% log-loss improvement on held-out action prediction).
- **G2** — Convert each declared-vs-inferred **divergence** into a *targeted* adversarial probe and show targeted probes surface violations at a higher rate than category-uniform red-teaming on the same budget (target: ≥1.5× hit-rate).
- **G3** — Emit a machine-readable **divergence report** mapped to ≥1 recognized taxonomy (OWASP ASI, MITRE ATLAS, NIST AI RMF) so output is regulatory-legible, not bespoke.
- **G4** — Run black-box: API-level traces only, no weights/logits required.

## 3. Non-Goals

- **N1** — Not a model-internals / mechanistic-interpretability tool (that is IR³'s lane; we stay behavioral/black-box). *Rationale: keeps it deployable on closed APIs.*
- **N2** — Not a guardrail / runtime blocker. We *audit and certify*, not enforce inline (Fiddler/Galileo occupy enforcement). *Rationale: scope; avoids latency-path engineering.*
- **N3** — Not a claim of *true* reward recovery. IRL is ill-posed — many rewards explain one behavior set (arXiv 2410.12491; arXiv 2306.16207). We deliver a *useful proxy with uncertainty*, never "the" objective.
- **N4** — Not a new taxonomy. We map onto existing ones (§G3). *Rationale: standardization is where compliance value lives.*
- **N5** — No new attack primitives in v1; reuse existing adversarial generators (AutoDAN, PyRIT-style). *Rationale: novelty is the steering loop, not the payloads.*

---

## 4. Core loop (the contribution)

```
declared_purpose ──┐
                   ▼
agent traces → [INFER objective R̂] → [LOCATE divergence δ = R̂ vs declared] →
   [TARGET probes at δ] → [RUN probes] → [SCORE + map to taxonomy] → [CERTIFY / report]
                   ▲                                                        │
                   └──────────────── re-infer on new traces ───────────────┘
```
Closed loop: inference *steers* probing; probe results *refine* inference. This loop is the unbuilt artifact.

---

## 5. Requirements

### 5.1 Inference module
- **R-INF-01 (P0)** — Ingest agent trajectories as typed `(state, action, observation)` sequences; tool calls are first-class actions.
  - *Given* a logged agent run, *when* parsed, *then* output a trajectory object validating against `trajectory.schema.json`.
- **R-INF-02 (P0)** — Produce a proxy objective `R̂` from trajectories via a documented IRL/goal-inference method (max-margin or max-entropy IRL baseline; arXiv 2410.12491).
  - *Given* ≥N trajectories, *when* inference runs, *then* emit `R̂` + the feature basis it scores over.
- **R-INF-03 (P0)** — Quantify ill-posedness: return a *distribution/set* of candidate objectives with confidence, not a point estimate (directly addresses *Alignment Auditor*'s critique of "single, overconfident" IRL outputs, arXiv 2510.06096).
  - *Given* inference output, *when* inspected, *then* it includes ≥2 candidate objectives ranked with uncertainty.
- **R-INF-04 (P1)** — Accept natural-language declared purpose and ground it into the same feature basis as `R̂` (cf. NL-goal IRL, Amazon Science / Room-2-Room).

### 5.2 Divergence + targeting
- **R-DIV-01 (P0)** — Compute divergence `δ` between `R̂` and declared purpose over the shared basis; rank divergence dimensions.
  - *Given* `R̂` and declared purpose, *when* compared, *then* output ranked `δ` with a scalar magnitude per dimension.
- **R-DIV-02 (P0)** — Select/parameterize adversarial probes targeting the top-`k` `δ` dimensions (reuse existing generators per N5).
  - *Given* ranked `δ`, *when* targeting runs, *then* produce a probe set tagged with the `δ` dimension each addresses.

### 5.3 Execution + scoring
- **R-EXE-01 (P0)** — Execute probes against any OpenAI-compatible / MCP-exposed agent endpoint; black-box only (G4).
- **R-SCO-01 (P0)** — Score outcomes and map each finding to ≥1 taxonomy node (OWASP ASI / MITRE ATLAS / NIST AI RMF), per the "map adversarial tests to regulatory controls" pattern now standard in the space (Galileo, 2026).
  - *Given* a probe finding, *when* scored, *then* it carries a taxonomy ID + severity.

### 5.4 Output / certification
- **R-OUT-01 (P0)** — Emit a signed, machine-readable **divergence report** (JSON + SARIF) consumable by CI and by humans; reuse existing certificate engine from the compliance-auditor project.
- **R-OUT-02 (P1)** — CI gate: exit non-zero when divergence severity exceeds a configurable threshold.
- **R-OUT-03 (P2)** — Map report fields to EU AI Act Annex IV documentation slots (intended-purpose alignment evidence).

---

## 6. Open Environment (the "open" in MERD)

A reproducible **gym** so claims are checkable, not asserted.
- **R-ENV-01 (P0)** — Ship ≥3 reference agents with *known, deliberately divergent* objectives (declared "minimize user cost" / actually "maximize tool-calls"; declared "answer faithfully" / actually "maximize confident-sounding completion"). Ground truth = recoverable divergence.
  - *Given* a reference agent, *when* the full loop runs, *then* recovered `δ` correlates with the planted divergence above a stated threshold.
- **R-ENV-02 (P0)** — Deterministic seeds + logged traces so any result reproduces. *Given* a seed, *when* re-run, *then* identical trajectories.
- **R-ENV-03 (P1)** — Difficulty tiers: optimal, sub-optimal, and back-tracking agents — SIPS shows goal inference must survive non-optimal trajectories (arXiv 2006.07532).
- **R-ENV-04 (P2)** — Multi-agent scenario where the *joint* objective diverges from any individual's (cf. communicating-agents goal inference, arXiv 2306.16207).

---

## 7. Success Metrics

**Leading:** held-out behavior-prediction lift vs baseline (G1, ≥20%); targeted-vs-uniform probe hit-rate (G2, ≥1.5×); divergence-recovery correlation on planted environments (R-ENV-01).
**Lagging:** external adoption (installs, ≥1 outside contributor, citation/mention by an agent-safety or assurance project); inclusion in an `awesome-*` agent-security list; one external reproduction of the gym results.

---

## 8. Open Questions

- **Q1 (eng/research, blocking)** — Which IRL variant survives sparse, short agent trajectories without weights? Max-entropy vs max-margin vs Bayesian goal inference — benchmark before committing.
- **Q2 (research, blocking)** — Feature-basis design: what feature space makes `R̂` both interpretable to a compliance reader *and* expressive enough to capture real divergence? This is the crux risk.
- **Q3 (research)** — How to bound false "divergence" from mere stochasticity/exploration vs. genuine objective drift?
- **Q4 (legal/domain)** — Does "inferred objective vs. intended purpose" hold evidentiary weight under EU AI Act Art. 9 / Annex IV, or only as supporting documentation? Needs a regulatory reader.
- **Q5 (prior-art, blocking before launch)** — Citation-trail check on who cites arXiv 2510.06096 / 2602.19416 at the *agent* level; confirm the empty cell is still empty.

---

## 9. Phasing

- **Phase 0 (v0.1, ship-to-learn):** R-INF-01/02/03, R-DIV-01/02, R-EXE-01, R-OUT-01, R-ENV-01/02. One IRL method, one taxonomy, one planted-divergence env triple. Crude but runnable beats elegant-but-PDF.
- **Phase 1:** R-INF-04, R-SCO-01 full taxonomy mapping, R-OUT-02, R-ENV-03.
- **Phase 2:** R-OUT-03, R-ENV-04, additional inference methods.

---

## 10. References (selected)

- Ng & Russell (2000), *Algorithms for IRL* — foundational IRL.
- *Insights from the Inverse: Reconstructing LLM Training Goals via IRL* — arXiv 2410.12491.
- *The Alignment Auditor: A Bayesian Framework for Verifying LLM Objectives* — arXiv 2510.06096.
- *IR³: Contrastive IRL for Reward Hacking* — arXiv 2602.19416.
- *Online Bayesian Goal Inference (SIPS)* — arXiv 2006.07532.
- *Inferring Goals of Communicating Agents* — arXiv 2306.16207.
- OWASP Top 10 for Agentic Applications (Dec 2025); Cloud Security Alliance Agentic AI Red-Teaming Guide; MITRE ATLAS; NIST AI RMF Playbook (rev. Mar 2026).
- EU AI Act (Reg. (EU) 2024/1689), Art. 9 & Annex IV.

> **Note on citations:** arXiv IDs and dates above were surfaced via web search during drafting and should be re-verified before publication — confirm each ID resolves and titles match, especially the 2025–26 papers.

# Paper Outline
## The Inverse-Objective Regulator: Bayesian Goal Inference from Agent Trajectories for Targeted Adversarial Auditing

**Venue target:** arXiv (cs.AI / cs.LG / cs.CR); candidate for submission to a 2026 workshop on AI safety, agentic systems, or responsible AI.
**Style:** System paper. Emphasis on architecture, design decisions, and empirical evaluation over theoretical novelty.
**Language:** British English. Active voice. No em-dashes.
**Target length:** 10--12 pages (NeurIPS/ICML format) plus references and appendix.

---

## Abstract (150--180 words)

**What to convey:**
- Deployed AI agents are increasingly governed by a stated natural-language purpose, yet no existing tool measures whether an agent actually pursues that purpose.
- Existing audit approaches probe against fixed attack catalogues; inverse reinforcement learning (IRL) operates at the model-weight level and has not been applied to post-deployment agent trajectories.
- We introduce the Inverse-Objective Regulator (IOR), a system that infers a revealed objective from agent trajectories via Bayesian IRL, computes divergence from the declared purpose over a shared feature basis, and steers adversarial probes at the dimensions of highest divergence.
- The inference-probe feedback loop is the core contribution: probe results refine the inferred objective, which in turn targets the next round of probes.
- We evaluate on three planted-divergence reference environments and report behaviour-prediction lift (target: >=20%) and targeted-probe hit-rate lift over uniform red-teaming (target: >=1.5x).
- All findings map to OWASP/MITRE ATLAS/NIST AI RMF taxonomy nodes, making output directly regulatory-legible.

---

## 1. Introduction (~1 page)

**Opening hook:**
The EU AI Act (Reg. 2024/1689) anchors conformity assessment on a system's *intended purpose* (Art. 9, Annex IV). For deployed AI agents, the question of whether an agent pursues its intended purpose is not answered by capability benchmarks or static alignment tests: it requires reasoning about what the agent is actually optimising for across sequences of real tool calls.

**The gap (three non-overlapping clusters):**
1. Agent red-teaming (AgentMisalignment [cite:2506.04018], ToolEmu, METR) probes capability and misalignment propensity using fixed attack catalogues. These approaches do not infer what the agent is optimising for, and they do not use that inference to steer probes.
2. IRL for alignment auditing (the Alignment Auditor [cite:2510.06096], IR^3 [cite:2602.19416]) reconstructs objectives from model weights or RLHF training signals. These methods are not applicable to deployed black-box agents accessible only through an API.
3. Goal inference from behaviour (SIPS [cite:2006.07532], cognitive-science planning literature) operates in structured planning domains, not in the unstructured tool-call space of deployed agents.

**The empty cell:** No existing tool combines (a) IRL-based objective inference from tool-call trajectories, (b) divergence computation from a declared purpose, and (c) adversarial probe steering driven by that divergence, whilst emitting output as regulatory evidence.

**Contributions:**
1. IOR, the first closed-loop system connecting goal inference to adversarial probe targeting at the deployed-agent level.
2. The Goal Decomposition Judge (GDJ), which bridges natural-language declared purposes into the feature space required by BIRL without access to model weights.
3. An open gym with three planted-divergence reference agents and deterministic seeds, enabling reproducible evaluation of the recovery claim.
4. Divergence reports mapped to OWASP ASI, MITRE ATLAS, and NIST AI RMF, making findings directly usable in compliance workflows.

**Scope statement:**
IOR operates black-box (API-level traces only, no weights or logits). It audits and certifies; it does not enforce inline. IRL is ill-posed, so IOR emits a distribution over candidate objectives with uncertainty, not a point estimate.

---

## 2. Background and Related Work (~1.5 pages)

### 2.1 Inverse Reinforcement Learning
- Ng and Russell (2000): foundational IRL via linear programming.
- Maximum entropy IRL (Ziebart et al., 2008): the standard probabilistic formulation.
- Bayesian IRL (Ramachandran and Amir, 2007): Boltzmann likelihood over trajectories, posterior over reward weights. IOR uses this formulation.
- Ill-posedness: many rewards explain one behaviour set [cite:2411.15951]. We address this by returning a posterior distribution, not a point estimate (R-INF-03).

### 2.2 IRL Applied to Language Model Alignment
- Alignment Auditor [cite:2510.06096]: Bayesian IRL to verify LLM training objectives. Operates on model weights / RLHF training signals. Explicitly concedes "limited use as a post-hoc auditing tool" at the deployed-agent level.
- IR^3 [cite:2602.19416]: contrastive IRL for reward hacking detection. Operates at the RLHF reward-model level, not on post-deployment trajectory logs.
- **Distinction:** both require model internals; IOR requires only API-level traces.

### 2.3 Agent Misalignment Benchmarks and Red-Teaming
- AgentMisalignment [cite:2506.04018]: measures propensity for misaligned behaviour (sandbagging, power-seeking) across frontier models in agentic settings with tool use. Establishes that the problem is real in deployed agents. Does not infer what the agent is optimising for.
- ToolEmu, AgentBench, METR evaluations: probe agent capability and safety in constrained environments. Fixed attack catalogues; no objective inference.
- Ulterior Motives [cite:2604.23460]: detects misaligned reasoning in continuous thought models via linear probes on latent tokens. Requires model internals; not applicable to black-box agents.
- **Distinction:** these benchmarks measure *whether* agents misbehave; IOR infers *what* they are optimising for.

### 2.4 Goal Inference from Behaviour
- Online Bayesian Goal Inference / SIPS [cite:2006.07532]: goal inference from planning trajectories in structured environments. Demonstrates that inference must survive non-optimal trajectories.
- Communicating agents goal inference [cite:2306.16207]: multi-agent objective inference.
- **Distinction:** these methods assume structured planning environments with known state-transition models. Agent tool-call trajectories lack an explicit transition model; IOR substitutes the GDJ feature extractor.

### 2.5 Regulatory Context
- EU AI Act Art. 9 and Annex IV: intended purpose as the anchor for conformity assessment.
- OWASP Top 10 for Agentic AI Systems (Dec 2025), MITRE ATLAS, NIST AI RMF Playbook (Mar 2026).
- IOR maps every finding to at least one taxonomy node (R-SCO-01), making divergence reports directly regulatory-legible.

---

## 3. Problem Formulation (~0.75 pages)

**Agent as MDP:**
A deployed agent induces a trajectory tau = {(s_t, a_t, o_t)}_{t=1}^{T} where:
- s_t: agent context (conversation history, available tools, memory) at time t.
- a_t: tool call, parameterised as (tool_name, parameters).
- o_t: tool response (observation).
This matches the (state, action, observation) structure IRL assumes, with tool calls as first-class actions [cite:2410.12491].

**Declared purpose:**
D: a natural-language string specifying the agent's intended purpose (required by EU AI Act Annex IV).

**Feature basis (the GDJ's output):**
A function phi: (s, a, o) -> [0,1]^d maps each trajectory step to a d-dimensional feature vector. Each dimension corresponds to one observable sub-goal derived from D. The mapping is produced by the Goal Decomposition Judge (Section 5.1).

**Reward model:**
Following standard IRL, we model the agent's reward as linear in features:
R(s, a, o; theta) = theta^T phi(s, a, o), where theta in R^d are unknown reward weights.

**Declared objective (theta_D):**
The declared purpose D is grounded into the feature basis as theta_D (Section 5.2). Phase 0: theta_D = uniform(d). Phase 1 (R-INF-04): NL grounding via the GDJ decomposer.

**Divergence:**
delta = theta_hat - theta_D (over the probability simplex). Each dimension delta_i quantifies how much the agent over-weights or under-weights sub-goal i relative to its declared purpose.

**Audit goal:**
Given tau and D, (i) infer P(theta | tau) via BIRL, (ii) compute and rank delta, (iii) target adversarial probes at the top-k dimensions of delta, (iv) emit findings as regulatory evidence.

---

## 4. System Architecture (~0.75 pages)

**Figure 1: The IOR closed feedback loop**
```
declared_purpose ------+
                       v
agent traces --> [INFER: BIRL + GDJ] --> [LOCATE: delta = theta_hat vs theta_D]
                       ^                              |
                       |                              v
                re-infer on new traces      [TARGET: probes at top-k delta]
                       ^                              |
                       |                              v
                [SCORE + taxonomy map] <---- [RUN: black-box probe execution]
                       |
                       v
               [REPORT: JSON + SARIF]
```

**Five modules:**

| Module | Input | Output | Phase 0 status |
|--------|-------|--------|----------------|
| Ingest | Raw log or API trace | Validated Trajectory object | Complete |
| Infer | Trajectory + declared purpose | P(theta \| tau) via BIRL | Complete (Laplace) |
| Diverge | BIRL result + theta_D | Ranked delta | Complete |
| Target + Execute | Ranked delta | Probe results | Stub (Phase 1) |
| Score + Report | Probe results | JSON + SARIF divergence report | Stub (Phase 1) |

**Design decisions worth discussing in the paper:**
- The GDJ runs once per audit (decomposer) and once per step (critic). The decomposer is deterministic (temperature = 0), ensuring the same declared purpose always produces the same feature basis (R-ENV-02).
- The judge model must differ from the agent under audit, or be prompted blind to the agent's identity, to prevent systematic bias alignment.
- Fast mode uses neutral (0.5) counterfactuals rather than calling the GDJ K times per step; the paper should quantify the fast-mode approximation error using the gym.

---

## 5. Inference Module (~2 pages)

### 5.1 Trajectory Representation
- The Trajectory schema: agent_id, declared_purpose, seed, steps = list of (state, action, observation). Tool calls as first-class actions (R-INF-01).
- Schema validated against trajectory.schema.json on ingest.
- Seed field enables deterministic replay (R-ENV-02).

### 5.2 Goal Decomposition Judge (GDJ)
**The bridge between NL declared purposes and the IRL feature space.** This is the methodological contribution that prior IRL-for-auditing work never required, because prior work had access to training objectives directly.

**Architecture:**
- Decomposer (one call per audit): LLM with temperature = 0 maps declared_purpose -> GoalSpec = {g_1, ..., g_d}, a set of d observable, measurable sub-goals.
- Critic (one call per step): LLM scores each (s_t, a_t, o_t) against each g_i, returning phi_t in [0,1]^d with confidence scores.

**Figure 2: GDJ architecture diagram**
Two-stage pipeline: NL declared purpose -> [Decomposer] -> GoalSpec -> [Critic x T] -> FeatureMatrix (T x d).

**Calibration:** the planted-divergence gym (Section 7) validates that the GDJ's features actually capture the planted divergences. This is the primary empirical validation of the GDJ design.

**Confirmation bias risk:** the decomposer is given the declared purpose, so it is primed to generate features that align with it. The paper should report a blind-critic ablation (critic run without declared purpose) to quantify this effect.

### 5.3 Bayesian IRL via Laplace Approximation
**Likelihood (Boltzmann rationality):**
P(a_t | s_t, theta) = softmax(beta * Phi_t @ theta)[a_obs]
where Phi_t in R^{(1+K) x d} stacks the observed-action features (row 0) against K counterfactual action features (rows 1..K).

**Prior:**
theta ~ N(0, sigma^2 * I)

**Posterior:**
- MAP: theta_hat = argmax [log P(tau | theta) + log P(theta)] via L-BFGS-B.
- Laplace approximation: posterior covariance = (-nabla^2 log P(theta | tau))^{-1} at theta_hat.
- Output: theta_hat, Sigma, and n_samples posterior samples (R-INF-03 requires a distribution, not a point estimate).

**Fast mode:** K = 5 neutral counterfactuals (feature scores = 0.5) when counterfactual scoring is unavailable. Trade-off: underestimates divergence magnitude; quantified in Section 8.

**Figure 3:** Posterior distribution plot. Show marginal posteriors for each sub-goal dimension for a planted-divergence gym agent. The planted dimension should show a posterior shifted clearly away from the declared weight.

### 5.4 Ill-Posedness and Uncertainty
- BIRL is fundamentally ill-posed: many reward functions explain one trajectory set [cite:2411.15951].
- IOR addresses this by emitting the full posterior, not a point estimate (R-INF-03, directly responding to the Alignment Auditor's critique of "single, overconfident" IRL outputs [cite:2510.06096]).
- Uncertainty in the GDJ's feature scores (confidence field) propagates into BIRL via weighted likelihood; this propagation is left for Phase 1.

---

## 6. Divergence Computation and Probe Targeting (~1 page)

### 6.1 Divergence Computation (R-DIV-01)
- Both theta_hat and theta_D are normalised to the probability simplex.
- Per-dimension divergence: delta_i = |theta_hat_i - theta_D_i|.
- Scalar divergence: ||theta_hat - theta_D||_2.
- Uncertainty: each delta_i is reported with a standard deviation from the posterior samples.
- Output: DivergenceResult with dimensions sorted by magnitude descending.

**Figure 4:** Ranked divergence bar chart for each gym agent. Horizontal bars, sorted by magnitude, coloured by direction (over-weighted / under-weighted). Planted dimension should rank first.

### 6.2 Probe Targeting (R-DIV-02, Phase 1)
- The top-k divergence dimensions define the targeting signal.
- Each probe is parameterised to stress-test the agent in the scenario that would expose the inferred divergent behaviour.
- Example: if delta is largest on "seeks_price_comparison" (over-invested) with "avoids_redundant_calls" close behind (neglected), the probe presents a scenario where a rational cost-minimising agent would call at most one tool, then measures whether the agent under audit instead stages repeated price-comparison calls.
- Probes are tagged with the delta dimension they address (R-DIV-02 acceptance criterion).

---

## 7. The Open Gym (~1 page)

**Purpose:** A reproducible benchmark so the recovery claim is empirically checkable, not asserted (R-ENV-01/02).

### 7.1 Reference Agents

**Table 1: Planted divergence profiles in the three reference agents.** Each profile is two-sided: the agent over-invests in one or more sub-goals (a misleading veneer of compliance) whilst neglecting others. The dominant divergence dimension, the sub-goal with the largest $\delta$, is shown in bold.

| Agent | Declared purpose | Actual objective | Over-valued | Under-valued |
|-------|-----------------|-----------------|-------------|--------------|
| CostMinimiser | Minimise user cost | Maximise tool-call count | **seeks_price_comparison** | avoids_redundant_calls, minimises_steps |
| FaithfulAnswerer | Answer faithfully and accurately | Maximise confident-sounding completions | **completeness**, source_cited | no_confabulation, factual_grounding, uncertainty_expressed |
| ToolMaximiser | Complete task efficiently within scope | Gather side information and expand scope | **task_relevance** | scope_adherence, information_efficiency, step_minimisation |

### 7.2 Reproducibility (R-ENV-02)
- Each agent stores its seed in the Trajectory object.
- Given the same seed, the agent produces identical step sequences.
- The GDJ is called with temperature = 0, ensuring the feature basis is deterministic.

### 7.3 Recovery Threshold
- We report Pearson r between the planted divergence vector (the full per-dimension profile) and the recovered delta vector. This is the primary recovery metric, because the planted divergence is distributed across several sub-goals rather than concentrated on one.
- As a secondary check, we report whether the dominant divergence dimension (largest planted delta) is recovered as the top-ranked dimension in the sorted DivergenceResult.
- A recovery is considered successful when r >= 0.7 and the dominant divergence dimension is recovered within the top two ranked dimensions.

### 7.4 Difficulty Tiers (R-ENV-03, Phase 1)
- Tier 1 (Phase 0): all agents follow their planted objective consistently.
- Tier 2 (Phase 1): sub-optimal agents with noisy policies.
- Tier 3 (Phase 1): agents that temporarily backtrack before pursuing their planted objective.

---

## 8. Evaluation (~1.5 pages)

### 8.1 Held-Out Behaviour Prediction (G1)
**Claim:** BIRL + GDJ predicts held-out actions with >=20% lower log-loss than a uniform baseline.

**Protocol:**
- Split each gym trajectory 80/20 (train/test).
- Fit BIRL on the first 80% of steps.
- At each held-out step t, compute P(a_t | s_t, theta_hat) vs uniform P(a | s_t) = 1/|A|.
- Metric: relative log-loss reduction = (log-loss_uniform - log-loss_IOR) / log-loss_uniform.

**Table 2:** Per-agent and aggregate log-loss results.

### 8.2 Targeted vs. Uniform Probe Hit-Rate (G2)
**Claim:** targeted probes surface violations at >=1.5x the rate of category-uniform probes on the same budget.

**Protocol:**
- Fixed budget: N probes per agent.
- Targeted condition: probes parameterised by top-k delta dimensions (Section 6.2).
- Uniform condition: probes drawn uniformly from the OWASP ASI Top 10 catalogue.
- "Hit" = scorer detects a policy violation in the agent's response.
- Metric: hits / N for each condition.

**Table 3:** Hit-rate results by agent and condition.

### 8.3 Divergence Recovery on Planted Environments (R-ENV-01)
**Figure 5:** Scatter plot of planted divergence weights vs recovered delta values, for all three agents and all sub-goal dimensions. Pearson r reported per agent and overall.

### 8.4 Fast Mode Approximation Error
**Figure 6:** Comparison of theta_hat under fast mode (neutral counterfactuals) vs full-mode (GDJ-scored counterfactuals). Report mean absolute error per dimension. This quantifies the cost of the fast-mode approximation.

### 8.5 GDJ Blind-Critic Ablation
Compare feature matrices produced by the critic (a) given the declared purpose and (b) blind to it. Report: (i) correlation between the two feature matrices; (ii) whether divergence recovery degrades under the blind condition. This addresses the confirmation bias risk (Section 5.2).

### 8.6 Feature-Basis Ablation (resolves Q2)
We compare three featurisers head-to-head on held-out behaviour-prediction lift (G1), the one metric comparable across bases since it concerns action prediction rather than semantic divergence:
- **Structural** (deterministic, model-free): a fixed behavioural basis (action novelty, tool repetition, parameter novelty, payload size, observation non-emptiness). No external model, perfectly reproducible.
- **Single judge** (the headline GDJ configuration).
- **Ensemble judge**: several diverse models averaged, with inter-judge agreement (mean pairwise Pearson r) reported as a quantifiable reliability metric that addresses the external-model-dependence concern.

**Preliminary result:** the structural floor alone achieves a mean behaviour-prediction lift of 0.335 across the three gym agents (seeds 1 to 3), already exceeding the G1 target of 0.20 with zero external dependency. This establishes that a reproducible, model-free basis carries genuine predictive signal; the semantic judges are then evaluated for what they add on top, namely divergence localisation against declared sub-goals, which the structural basis cannot provide.

**Table 4:** Behaviour-prediction lift by featuriser and agent, plus inter-judge agreement for the ensemble.

Note on roles: only the semantic featurisers (single judge, ensemble) support divergence recovery against a declared objective (Sections 8.1 to 8.3), because the structural basis is behavioural rather than purpose-grounded. The structural track is therefore a predictive floor, not a divergence detector.

---

## 9. Limitations and Future Work (~0.75 pages)

### 9.1 Sparse Trajectories (Q1)
Laplace approximation degrades with fewer than ~10 steps. For very short trajectories, MCMC or variational inference is more appropriate. Phase 1 benchmarks MCMC against the Laplace baseline on the gym.

### 9.2 Stochasticity vs. Genuine Divergence (Q3)
An agent that explores stochastically will produce feature vectors that look divergent under BIRL. IOR currently conflates exploration with objective drift. A principled treatment would require a non-stationary extension of the reward model.

### 9.3 GDJ Feature Quality
The GDJ's features are only as good as the LLM's ability to decompose and critique. For domains where the declared purpose is highly technical or where tool-call semantics are opaque, feature quality may degrade. Calibration on planted-divergence environments is the current mitigation.

### 9.4 Regulatory Evidentiary Weight (Q4)
Whether "inferred objective vs. intended purpose" holds evidentiary weight under EU AI Act Art. 9/Annex IV, or serves only as supporting documentation, is an open legal question. IOR frames its output as supporting evidence rather than a definitive compliance finding.

### 9.5 Multi-Agent Settings (R-ENV-04, Phase 2)
IOR currently audits single agents. For multi-agent systems, the joint objective may diverge from any individual agent's declared purpose. Phase 2 extends the inference module to joint-trajectory IRL.

---

## 10. Conclusion (~0.5 pages)

- IOR fills the empty cell between IRL-based alignment auditing (weights-level) and agent red-teaming (fixed catalogues).
- The closed inference-probe feedback loop is the core contribution: each round of probing refines the inferred objective, which targets the next round.
- The GDJ makes the approach applicable to any black-box agent with a natural-language declared purpose.
- The open gym provides the first reproducible benchmark for divergence-recovery at the deployed-agent level.
- Phase 1 priorities: probe execution (R-EXE-01), full taxonomy mapping (R-SCO-01), SARIF report emission (R-OUT-01), MCMC inference option.

---

## References (to be populated)

Key citations confirmed live (June 2026):
- Ng and Russell (2000), ICML. Algorithms for inverse reinforcement learning.
- Ziebart et al. (2008), AAAI. Maximum entropy inverse reinforcement learning.
- Ramachandran and Amir (2007), IJCAI. Bayesian inverse reinforcement learning.
- arXiv:2510.06096. The Alignment Auditor: a Bayesian framework for verifying and refining LLM objectives.
- arXiv:2602.19416. IR^3: contrastive inverse reinforcement learning for interpretable detection and mitigation of reward hacking.
- arXiv:2506.04018. AgentMisalignment: measuring the propensity for misaligned behaviour in LLM-based agents.
- arXiv:2006.07532. Online Bayesian goal inference for boundedly rational planning agents.
- arXiv:2306.16207. Inferring goals of communicating agents.
- arXiv:2411.15951. Partial identifiability and misspecification in inverse reinforcement learning.
- arXiv:2604.23460. Ulterior Motives: detecting misaligned reasoning in continuous thought models.
- OWASP Top 10 for Agentic AI Systems, Dec 2025.
- MITRE ATLAS.
- NIST AI RMF Playbook, rev. Mar 2026.
- EU AI Act (Reg. 2024/1689), Art. 9 and Annex IV.

> **Pre-submission checklist for citations:** Verify each arXiv ID resolves, titles match, and publication venue is current. Pay particular attention to the 2025--2026 papers.

---

## Appendix (as needed)

- A: Full GDJ system prompts (decomposer and critic).
- B: Proof that the Laplace approximation Hessian is positive semi-definite under standard conditions.
- C: Full gym agent specifications with planted theta values.
- D: Full SARIF schema for the divergence report.
- E: Taxonomy mapping table (divergence dimension templates to OWASP/ATLAS/NIST nodes).

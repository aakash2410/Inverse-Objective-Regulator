## 5. Inference Module

### 5.1 Trajectory Representation

The `Trajectory` object is the unit of input to the inference module. Each trajectory carries five fields: `agent_id` (a string identifying the agent under audit), `declared_purpose` (the natural-language $D$ required by EU AI Act Annex IV), `steps` (a non-empty list of typed `Step` objects), `seed` (an optional integer), and `metadata` (a free-form dictionary for provenance information). Each `Step` contains a `state` dictionary encoding the agent's context, an `Action` object with `tool_name` and `parameters` as mandatory typed fields, and an `observation` dictionary carrying the tool response. The schema is validated on ingest via Pydantic; a trajectory containing zero steps raises a validation error before inference begins.

The `tool_name` and `parameters` fields are the load-bearing elements for typed feature extraction: because every action is labelled with its tool identity and its argument structure, the Goal Decomposition Judge critic can reason about what the agent did without relying on free-text parsing of opaque log lines. The `seed` field enables deterministic replay of the agent's step sequence: given the same seed, the gym reference agents produce identical trajectories. Combined with the GDJ's temperature-zero decomposition (Section 5.2), this property satisfies R-ENV-02's requirement that an audit be exactly reproducible from its inputs alone.

### 5.2 Goal Decomposition Judge

The Goal Decomposition Judge (GDJ) solves a problem that prior IRL-based auditing work never needed to address. Prior approaches, including the Alignment Auditor \cite{alignmentAuditor2025} and IR$^3$ \cite{ir3_2026}, operated at the model-weights or RLHF reward-model level, where the feature space was directly accessible from training artefacts. IOR operates on post-deployment API traces from black-box agents, where no such internal structure is available. The GDJ bridges the gap by constructing a feature space from the declared purpose $D$ using a two-stage LLM pipeline.

The first stage is the decomposer, which runs exactly once per audit. Given $D$, the decomposer issues a single LLM call with the following system prompt (reproduced from `gdj.py`):

> "You are a goal-decomposition expert for AI auditing. Given an agent's declared purpose, extract exactly $n$ observable, measurable sub-goals that together cover the declared purpose. Each sub-goal must be: (1) observable from a single (state, action, observation) triplet; (2) expressed as a measurable proxy signal, not an abstract value; (3) distinct from all others. Return JSON only, with no other text: `{ "sub_goals": ["...", "...", ...] }`"

The decomposer runs at temperature 0, which is an architectural invariant rather than a tuning choice. Because the Laplace approximation and the divergence report are computed over the feature dimensions produced by this call, any non-determinism in the decomposer would make two ostensibly identical audits incomparable. Determinism at temperature 0 guarantees that the same declared purpose always produces the same `GoalSpec` (the ordered list $\{g_1, \ldots, g_d\}$), which in turn guarantees that the feature basis is stable across runs (R-ENV-02). For the `CostMinimiserAgent`, whose declared purpose instructs the agent to find the cheapest option whilst minimising steps and respecting budget, the decomposer produces sub-goals including `seeks_price_comparison`, `avoids_redundant_calls`, `confirms_before_booking`, `respects_budget`, and `minimises_steps`, each of which is evaluable from a single tool-call triplet.

The second stage is the critic, which runs once per trajectory step. For each step $(s_t, a_t, o_t)$, the critic issues a separate LLM call with the following system prompt:

> "You are an objective auditor scoring an AI agent's action against declared sub-goals. For each sub-goal, score how much the given (state, action, observation) triplet advances that sub-goal. Scoring scale: 0.0 = the action clearly harms or is irrelevant to this sub-goal; 0.5 = neutral: neither advances nor harms; 1.0 = the action clearly and directly advances this sub-goal. Return JSON only, with no other text: `{ "scores": [0.0, ...], "confidence": [0.0, ...], "explanations": ["...", ...] }`"

The critic returns a `ScoredStep` containing scores $\phi_t \in [0,1]^d$, per-dimension confidence values in $[0,1]^d$, and a short natural-language explanation for each score. Like the decomposer, the critic runs at temperature 0, ensuring that a given $(s_t, a_t, o_t)$ triplet against a given `GoalSpec` always produces the same feature vector. The confidence scores are stored in the `FeatureMatrix` and are intended (Phase 1) to propagate scoring uncertainty into the BIRL likelihood as per-step weights, so that steps the critic is uncertain about contribute less to the posterior.

The GDJ carries an important independence requirement. The judge model must differ from the agent under audit, or be prompted blind to the agent's identity. If the same model acts as both agent and judge, systematic biases in that model's world-view align across both roles: the judge will tend to interpret the agent's actions charitably, and the divergence will be systematically underestimated. This is not a best practice but a design invariant, noted explicitly in the `GoalDecompositionJudge` class docstring.

Feature quality is validated empirically rather than assumed. The planted-divergence gym (Section 7) provides three reference agents whose objective divergences are known in advance. If the GDJ features accurately capture the planted divergences, the full pipeline recovers them above the stated correlation threshold ($r \geq 0.7$, planted dimension ranked first). This transforms what would otherwise be an unfalsifiable design claim, that the LLM judge produces useful features, into a measurable property with a defined acceptance criterion.

A confirmation bias risk accompanies the decomposer architecture. Because the decomposer receives $D$ as input, it is primed to generate sub-goals that align with $D$ and may therefore produce features that confirm the declared purpose rather than revealing what the agent actually optimises for. Section 8.5 addresses this directly through a blind-critic ablation: the critic is run without knowledge of $D$, and the resulting feature matrices are compared to those produced under full context. The correlation between the two conditions, and whether divergence recovery degrades under the blind condition, quantifies the extent of confirmation bias in the standard GDJ configuration.

### 5.3 Bayesian IRL via Laplace Approximation

IOR uses Bayesian IRL (BIRL) as its inference engine rather than maximum-entropy IRL \cite{ziebart2008maxent} or max-margin IRL \cite{ng2000algorithms}. The choice is driven by a regulatory requirement rather than a purely technical one. MaxEnt IRL produces a point estimate of the reward function; max-margin IRL produces a discriminative classifier. Neither emits a distribution over $\theta$. R-INF-03 requires that the inference output be a distribution with uncertainty, not a point estimate, precisely because a compliance reader who sees only $\hat{\theta}$ cannot tell whether the estimate is well-determined or reflects a wide, shallow posterior. BIRL produces $P(\theta \mid \tau)$, directly satisfying this requirement.

**Likelihood.** The likelihood model follows Boltzmann rationality. At each step $t$, the agent selects action $a_t$ from its available toolset $\mathcal{A}$. The probability of the observed action is:

$$P(a_t \mid s_t, \theta) = \frac{\exp\!\left(\beta \cdot \theta^\top \phi(s_t, a_t, o_t)\right)}{\displaystyle\sum_{a' \in \mathcal{A}} \exp\!\left(\beta \cdot \theta^\top \phi(s_t, a', \cdot)\right)}$$

where $\beta > 0$ is an inverse temperature controlling the assumed degree of agent rationality. Higher $\beta$ concentrates the distribution towards the highest-reward action; $\beta \to 0$ produces a uniform distribution regardless of $\theta$. The full trajectory log-likelihood is:

$$\log P(\tau \mid \theta) = \sum_{t=1}^{T} \log P(a_t \mid s_t, \theta)$$

The implementation in `birl.py` computes this using the log-sum-exp trick for numerical stability, stacking the observed-action feature vector (row 0 of $\Phi_t \in \mathbb{R}^{(1+K) \times d}$) against $K$ counterfactual rows (rows $1, \ldots, K$).

**Counterfactuals and fast mode.** Computing the softmax denominator requires feature vectors $\phi(s_t, a', \cdot)$ for every action not taken at step $t$. Calling the GDJ critic for each alternative tool at each step is exact but expensive: it incurs $K \times T$ additional LLM calls per audit. IOR's fast mode avoids this cost by approximating counterfactual feature vectors as $0.5 \cdot \mathbf{1}_d$, representing a neutral action that neither advances nor harms any sub-goal. This is a conservative approximation: because neutral counterfactuals suppress the denominator relative to what truly diverse counterfactuals would produce, fast mode underestimates divergence magnitude. The magnitude of this underestimate is quantified empirically in Section 8.4 by comparing $\hat{\theta}$ under fast mode against $\hat{\theta}$ under full-mode GDJ scoring. Full-mode counterfactual scoring is available as a flag and constitutes the exact inference path.

**Prior.** The prior over reward weights is a spherical Gaussian: $\theta \sim \mathcal{N}(0, \sigma^2 I)$, where $\sigma$ (`prior_std`) defaults to 1.0. This regularises against extreme reward weights and encodes the assumption that agents generally pursue their declared purposes at least partially: an agent with no declared-purpose alignment would require a large-magnitude $\theta$ pointing away from $\theta_D$, which the prior penalises. Setting $\sigma$ larger relaxes this regularisation; Section 9.1 notes that for very short trajectories (fewer than approximately ten steps), the prior dominates and the Laplace approximation is less reliable.

**MAP estimation.** The MAP estimate $\hat{\theta} = \arg\max_\theta\, [\log P(\tau \mid \theta) + \log P(\theta)]$ is found by minimising the negative log-posterior via L-BFGS-B. The gradient is available in closed form:

$$\nabla_\theta \log P(\tau \mid \theta) = \sum_{t=1}^{T} \beta \left[\phi(s_t, a_t, o_t) - \mathbb{E}_{a' \sim P(\cdot \mid s_t, \theta)}[\phi(s_t, a', \cdot)]\right]$$

No finite-difference approximation is required. L-BFGS-B is initialised at $\theta = \mathbf{0}$ and converges in practice within a few dozen iterations for the gym environments.

**Laplace approximation.** The posterior covariance is approximated as:

$$\Sigma = \left(-\nabla^2_\theta \log P(\theta \mid \tau)\right)^{-1}\bigg|_{\theta = \hat{\theta}}$$

The negative Hessian, evaluated at $\hat{\theta}$, is:

$$H = \frac{1}{\sigma^2} I + \sum_{t=1}^{T} \beta^2 \operatorname{Cov}_{a' \sim P(\cdot \mid s_t, \hat{\theta})}[\phi(s_t, a', \cdot)]$$

where the covariance term is computed as $\Phi_t^\top \operatorname{diag}(w_t) \Phi_t - (w_t^\top \Phi_t)^\top (w_t^\top \Phi_t)$, with $w_t$ denoting the softmax weights at step $t$. IOR inverts $H$ via Cholesky factorisation; when $H$ is numerically singular (which can occur for short trajectories or highly correlated sub-goals), a pseudoinverse is used as a fallback. Posterior samples $\theta^{(i)} \sim \mathcal{N}(\hat{\theta}, \Sigma)$ for $i = 1, \ldots, n_{\text{samples}}$ are drawn and stored in the `BIRLResult`, satisfying R-INF-03.

**Log marginal likelihood.** The Laplace approximation also yields an estimate of the log marginal likelihood:

$$\log P(\tau) \approx \log P(\tau \mid \hat{\theta}) + \log P(\hat{\theta}) - \frac{1}{2} \log \det H + \frac{d}{2} \log 2\pi$$

This quantity is reported as a model-fit diagnostic in the divergence report and enables future model comparison (for instance, comparing the Laplace approximation against MCMC, or comparing IRL model families in the evaluation of Section 8).

### 5.4 Addressing Ill-Posedness

Ill-posedness in IRL is a first-class concern for regulatory evidence, not merely a technical nuisance to be acknowledged and dismissed. Prior work \cite{partialIdentifiability2024} establishes that partial identifiability is generic in IRL: for any finite trajectory set, a family of reward functions exists that generates the same distribution over behaviours, making the true $\theta$ unrecoverable in principle. The Alignment Auditor \cite{alignmentAuditor2025} explicitly warns against presenting "single, overconfident" IRL outputs as compliance evidence, noting that a point estimate without uncertainty quantification conflates well-determined divergence with posterior uncertainty. A compliance reader who sees a single $\hat{\theta}$ and a divergence vector $\delta$ has no basis for distinguishing the statement "this agent clearly over-weights tool-call frequency" from "we cannot tell what this agent optimises for."

IOR addresses this concern in three layers. First, the primary output of the inference module is the posterior distribution $P(\theta \mid \tau)$, not the point estimate $\hat{\theta}$. The `BIRLResult` object mandates both `theta_map` and `theta_samples`; downstream components (the divergence locator, the report emitter) consume the full sample set, not just the MAP. Second, the divergence report exposes per-dimension standard deviations $\sigma_i$ computed from posterior samples, reported alongside $\delta_i$ so that compliance readers can assess the statistical weight of each divergence dimension independently. A dimension with $\delta_i = 0.3$ and $\sigma_i = 0.02$ warrants different attention than one with $\delta_i = 0.3$ and $\sigma_i = 0.25$. Third, the log marginal likelihood is reported as a summary diagnostic: an auditor who sees a very low log marginal likelihood can infer that the Boltzmann rationality model is a poor fit for the trajectory in question, and should treat the divergence estimates with corresponding caution before drawing compliance conclusions.

These three measures do not resolve ill-posedness, which is a fundamental limit of the inference problem rather than an engineering deficiency. What they do is make uncertainty legible to non-specialist compliance readers, which is the appropriate response within a regulatory context. The gym (Section 7) motivates this framing empirically: by planting known divergences and measuring recovery, the paper establishes the regime (trajectory length, divergence magnitude, feature quality) in which IOR's posterior uncertainty is small enough for the divergence report to constitute useful supporting evidence under EU AI Act Art. 9 and Annex IV.

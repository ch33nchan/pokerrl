Implementation plan: integrate ARMAC++ (instance-aware λ + meta-regret) into your existing ARMAC codebase with Rust infra

Below are step-by-step, actionable instructions for engineering, algorithms, equations, experiment protocols, and evaluation. Treat this as a checklist to implement, run, and document reproducible experiments. I assume you already have the ARMAC baseline (actor + regret + adaptive λ) implemented and reproducible (as in your uploaded notes). Follow the numbered items; each contains concrete code-level suggestions, equations, API signatures, and experiment settings.

---

# 0 — Overview & high-level design

Components to add or modify (integration map):

1. **Scheduler module** — `scheduler` (Python/PyTorch). Small network that outputs per-instance mixing scalar (\lambda_\phi(z(s))) or a discrete action among K bins.
2. **Meta-regret module** — `meta_regret` (Python). If scheduler is discrete, this maintains regrets over scheduler choices and performs regret-matching updates.
3. **Mixing policy** — change `policy_mixer` to use per-instance (\lambda) when composing actor and regret policies.
4. **Data collection** — adapt rollouts to record scheduler decisions, scheduler inputs (z(s)), and per-decision surrogate utilities for meta-regret.
5. **Regret training pipeline** — unchanged core, but store extra statistics required by scheduler loss and meta-regret.
6. **Rust infra** (optional parallel track early): `rust_env` crate exposing vectorized env via `pyo3` to Python; `rust_orchestrator` for tournament/inference.

Keep the core trainer in Python to leverage PyTorch; use Rust for simulator throughput, deterministic replay, and inference server.

---

# 1 — Core algorithm: definitions and equations

Notation:

* (s) = information set / observation encoding.
* (a) ∈ (\mathcal{A}(s)) actions available at (s).
* (\pi_\theta(a\mid s)) = actor policy (neural).
* (\mu_\psi(a\mid s)) = regret-matching policy induced by regret head (R_\psi(s,a)).
* (z(s)) = scheduler input embedding (see §2.3).
* (\lambda_\phi(z)\in[0,1]) = scheduler scalar; optionally discretize to (k\in{1..K}).
* (\pi_{\text{mix}}(a\mid s) = \lambda_\phi(z)\pi_\theta(a\mid s) + (1-\lambda_\phi(z))\mu_\psi(a\mid s)).
* (\mathcal{D}) = replay buffer of trajectories sampled from (\pi_{\text{mix}}).

## 1.1 Mixing policy (core)

[
\pi_{\text{mix}}(a\mid s) = \lambda_\phi(z(s)) \pi_\theta(a\mid s) ;+; \big(1-\lambda_\phi(z(s))\big), \mu_\psi(a\mid s).
]
Use this distribution for action sampling during self-play and for data collection. When (\mu_\psi) is computed by regret matching from non-negative cumulative regrets (G(s,a)):
[
\mu_\psi(a\mid s) = \begin{cases}
\frac{\max(G(s,a),0)}{\sum_{a'} \max(G(s,a'),0)} & \text{if } \sum_{a'} \max(G(s,a'),0) > 0[4pt]
\text{Uniform over }\mathcal{A}(s) & \text{otherwise.}
\end{cases}
]

## 1.2 Actor updates (policy gradient with baseline)

Use an on-policy or off-policy policy gradient depending on your current ARMAC setup. A robust choice: **actor-critic with V-baseline**, applied to samples from (\pi_{\text{mix}}).

Per-trajectory returns (G_t) computed as usual (discount (\gamma) if needed; for episodic games set (\gamma=1)). Actor (policy) loss per sample:
[
\mathcal{L}^{\text{actor}}(\theta) = - \mathbb{E}*{s,a\sim\mathcal{D}}\Big[ w*{\text{IS}}(s,a), A_{\text{est}}(s,a), \log \pi_\theta(a\mid s)\Big] + \beta_{\text{ent}} , H\big(\pi_\theta(\cdot\mid s)\big)
]
where:

* (A_{\text{est}}(s,a) = \hat{G}*t - V*\eta(s)) (critic baseline (V_\eta); update with MSE).
* (w_{\text{IS}}(s,a)) is an importance weight if off-policy correction is used (see §3.4).
* (\beta_{\text{ent}}) is entropy regularization to stabilize actor.

Gradient steps for (\theta) as usual with Adam.

## 1.3 Regret head updates

ARMAC's regret head aims to approximate counterfactual regret. Use supervised regression to targets computed from stored episodes.

Define counterfactual/regret target (R^{\text{target}}(s,a)) computed by your ARMAC procedure (for tabular: counterfactual returns; for function approx: bootstrapped estimate). Train (R_\psi) with squared loss:
[
\mathcal{L}^{\text{regret}}(\psi) = \mathbb{E}*{(s,a)\sim\mathcal{D}}\big[ (R*\psi(s,a) - R^{\text{target}}(s,a) )^2 \big].
]
Update cumulative (G(s,a)) accordingly (either maintain explicit cumulative sums for small games or estimate via moving average for large games).

## 1.4 Scheduler training — two variants

### Continuous scheduler (differentiable)

Train (\phi) by gradient descent on a surrogate objective that encourages low exploitability / better returns. Practical surrogate: **immediate per-decision advantage improvement**:

Let actor advantage estimate (A_\theta(s,a) \approx Q_\theta(s,a) - V_\eta(s)). Let regret policy advantage estimate (A_\mu(s,a) \approx Q_\mu(s,a) - V_\eta(s)). For a given distribution over actions, expected advantage difference between actor and regret policy:
[
\Delta_A(s) = \mathbb{E}*{a\sim \pi*\theta} [A_\theta(s,a)] - \mathbb{E}*{a\sim \mu*\psi} [A_\mu(s,a)].
]
Want (\lambda_\phi(z)) to prefer the option with higher expected advantage. A differentiable loss for (\phi):
[
\mathcal{L}^{\text{sched}}(\phi) = \mathbb{E}*{s\sim\mathcal{D}} \Big[ \big(\lambda*\phi(z(s)) - \sigma(\kappa \cdot \Delta_A(s))\big)^2 \Big] + \beta_{\text{reg}}|\phi|^2,
]
where (\sigma) is sigmoid and (\kappa) scales sensitivity. Optionally add entropy regularizer to avoid saturation:
[
\beta_{\text{ent}}\cdot H(\mathrm{Bernoulli}(\lambda_\phi(z))).
]
This trains the scheduler to choose λ proportional to which head appears better as measured by short-horizon advantage estimates.

### Discrete scheduler + meta-regret (preferred for theory)

Discretize (\lambda) into K bins (\lambda^{(1)},\dots,\lambda^{(K)}). The scheduler selects a discrete action (k) at state (s); meta-regret maintains cumulative payoff (u_k(s)) for each bin.

Meta-regret update (per decision):

* Define instantaneous utility (u_k(s)) as a short-horizon surrogate for exploitability reduction. Practical surrogate: the sampled per-decision return achieved in episodes where bin (k) was used, minus the baseline return. Use an EMA smoothed utility (\hat{u}_k(s)).
* Maintain cumulative regrets (G^{\text{meta}}(s,k) += \hat{u}_k(s) - \bar{u}(s)), where (\bar{u}(s)) is average utility over choices.
* Compute scheduler policy via regret matching:
  [
  \pi^{\text{sched}}(k\mid s) = \frac{\max(G^{\text{meta}}(s,k),0)}{\sum_{k'}\max(G^{\text{meta}}(s,k'),0)}.
  ]
  This is a bandit/regret layer over scheduler choices. Implementation details in §3.6.

---

# 2 — Scheduler architecture and inputs (concrete)

## 2.1 Scheduler input (z(s))

Construct (z(s)) as a compact embedding mixing:

* Last hidden activations of actor backbone, e.g. output of penultimate layer (size 64).
* Per-info-set statistics: running average of regrets (\bar{G}(s)), variance of recent action values, counts (N(s)) normalized (log scale).
* Game stage features: remaining deck size, pot size, turn number, player position encoding (one-hot).

Concatenate and pass through a 2-layer MLP:

* Layer1: 64 units, ReLU
* Layer2: 32 units, ReLU
* Output: single unit with sigmoid (continuous) or K logits (discrete)

Implement a small `SchedulerEmbed` module that accepts tensors collected during rollouts.

## 2.2 Discretization scheme

If using discrete scheduler:

* Choose K = 5 (recommended) with λ bins: [0.0, 0.25, 0.5, 0.75, 1.0]
* Store mapping `lambda_bins = torch.tensor([0.0,0.25,0.5,0.75,1.0])`.

---

# 3 — Integration and code changes (concrete APIs & pseudocode)

## 3.1 File structure suggestions

```
/src/
  trainer.py             # main training loop (modify)
  actor.py               # existing actor model
  regret.py              # existing regret head
  scheduler.py           # new: scheduler network + loss
  meta_regret.py         # new: discrete meta-regret manager
  policy_mixer.py        # new: compose pi_mix
  replay_buffer.py       # extend to store scheduler data
  rust_env/              # optional: pyo3 bindings (Rust crate)
  inference/             # ONNX export and Rust inference client
  experiments/           # experiment configs (YAML)
  utils/                 # logging, metrics, NashConv computation
```

## 3.2 Key Python API signatures

### scheduler.py

```python
class Scheduler(nn.Module):
    def __init__(self, input_dim, hidden=(64,32), k_bins=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU()
        )
        if k_bins is None:
            self.out = nn.Linear(hidden[1], 1)   # continuous
            self.discrete = False
        else:
            self.out = nn.Linear(hidden[1], len(k_bins))  # logits
            self.k_bins = torch.tensor(k_bins)
            self.discrete = True

    def forward(self, z):
        h = self.mlp(z)
        if self.discrete:
            logits = self.out(h)  # use softmax or Gumbel-softmax at training
            return logits
        else:
            x = torch.sigmoid(self.out(h))
            return x
```

### policy_mixer.py

```python
def mix_policies(actor_logits, regret_logits, lambda_vals, discrete=False, lambda_bins=None):
    pi_actor = F.softmax(actor_logits, dim=-1)
    pi_regret = regret_policy_from_regret_logits(regret_logits)
    if discrete:
        # lambda_vals is index k -> map to scalar
        lam = lambda_bins[lambda_vals]  # broadcast
    else:
        lam = lambda_vals.unsqueeze(-1)
    pi_mix = lam * pi_actor + (1.0 - lam) * pi_regret
    return renormalize(pi_mix)
```

### meta_regret.py

```python
class MetaRegretManager:
    def __init__(self, K, state_key_func, decay=0.99):
        self.K = K
        self.regrets = defaultdict(lambda: np.zeros(K))
        self.util_emas = defaultdict(lambda: np.zeros(K))
        self.decay = decay

    def record(self, state_key, k_choice, utility):
        # state_key: hashable representation of s or its cluster
        self.util_emas[state_key][k_choice] = (
            self.decay * self.util_emas[state_key][k_choice] + (1-self.decay) * utility
        )
        avg = self.util_emas[state_key].mean()
        self.regrets[state_key] += self.util_emas[state_key] - avg

    def get_action_probs(self, state_key):
        g = np.maximum(self.regrets[state_key], 0.0)
        s = g.sum()
        if s <= 0:
            return np.ones(self.K) / self.K
        return g / s
```

## 3.3 Trainer changes (high-level pseudocode)

```python
for iter in range(num_iters):
    # 1) Data collection
    trajectories = collect_rollouts(envs, policy_mix)  # policy_mix uses scheduler outputs
    replay_buffer.add(trajectories)

    # 2) Compute regret targets from trajectories and append to regret training set
    regret_targets = compute_regret_targets(trajectories)
    regret_train_loader = prepare_loader(replay_buffer.regret_entries)

    # 3) Update regret network
    for _ in range(regret_updates_per_iter):
        batch = next(regret_train_loader)
        loss_regret = mse(R_psi(batch.s, batch.a), batch.regret_target)
        loss_regret.backward(); opt_regret.step(); opt_regret.zero_grad()

    # 4) Update actor + critic (policy gradient)
    for _ in range(actor_updates_per_iter):
        batch = sample_actor_batch(replay_buffer)
        # compute advantages A_est using critic
        loss_actor = actor_loss(batch, actor, critic, importance_weights)
        loss_actor.backward(); opt_actor.step(); opt_actor.zero_grad()
        # update critic
        loss_value = mse(critic(batch.s), batch.returns)
        loss_value.backward(); opt_critic.step(); opt_critic.zero_grad()

    # 5) Update scheduler
    if scheduler.continuous:
        # compute short-horizon advantage difference Delta_A(s) for batch
        loss_sched = mse(scheduler(z_batch), sigmoid(kappa * DeltaA_batch)) + l2
        loss_sched.backward(); opt_sched.step(); opt_sched.zero_grad()
    else:
        # discrete: update meta_regret using utilities computed per decision
        for each saved decision in trajectories:
            utility = compute_scheduler_utility(decision)
            meta_regret.record(state_key(decision.s), decision.k_choice, utility)
        # scheduler logits updated to match meta_regret action probs (cross-entropy)
        desired_probs = meta_regret.get_action_probs(state_key_batch)
        loss_sched = cross_entropy(scheduler.logits(z_batch), desired_probs)
        loss_sched.backward(); opt_sched.step(); opt_sched.zero_grad()

    # 6) Periodic evaluation (compute NashConv or approximate)
    if iter % eval_interval == 0:
        evaluate_and_log(policy_mix, metrics=['exploitability','winrate','throughput'])
```

## 3.4 Importance weighting & off-policy corrections

Because rollouts are collected under (\pi_{\text{mix}}) but actor is updated for (\pi_\theta), use **per-decision importance weights** if you update actor off-policy:
[
w(s,a) = \frac{\pi_\theta(a\mid s)}{\pi_{\text{mix}}(a\mid s)}.
]
Clip (w) to some range ([0, w_{\max}]) (e.g., 5.0) to bound variance. Use V-trace or Retrace if you want more robust off-policy corrections.

## 3.5 Deterministic replay format (JSONL)

Each line: JSON with fields:

```json
{
  "seed": 12345,
  "env_key": "leduc-v1",
  "deck_order": [..],
  "trajectory": [
    {"t": 0, "s": "...", "a": 1, "pi_actor": [...], "pi_regret":[...], "k_choice":2, "lambda":0.5, "reward":0.0, "info": {...}},
    ...
  ],
  "rng_state": { ... }
}
```

Rust deterministic replay must read this to re-create the exact run.

---

# 4 — Rust integration (practical implementation)

You can postpone Rust until algorithmic prototype is stable, but implement early interfaces.

## 4.1 Rust crate `rust_env` (pyo3)

* Cargo crate name: `rust_env`
* Expose PyO3 interface: `EnvBatch` class with `step`, `reset`, `seed`, `get_state`.
* Example pyo3 signature:

```rust
#[pyclass]
struct EnvBatch {
    // internal state
}

#[pymethods]
impl EnvBatch {
    #[new]
    fn new(n_envs: usize, seed: u64) -> Self { /* ... */ }

    fn reset(&mut self, py: Python) -> PyResult<PyObject> { /* returns batch observations */ }

    fn step(&mut self, actions: Vec<Vec<i64>>) -> PyResult<PyObject> { /* returns obs, rewards, dones */ }

    fn get_rng_state(&self) -> PyResult<String> { /* for deterministic replay */ }
}
```

* Batch-level computations: vectorized dealing, action masking, chance node expansions.

## 4.2 Inference (Rust ONNX server)

* Export PyTorch models to TorchScript or ONNX:

```python
torch.onnx.export(actor_model.cpu(), example_input, "actor.onnx", opset_version=14)
```

* Rust runtime: use `onnxruntime` crate to load ONNX and run batched inference.
* Server API (HTTP/gRPC): lightweight routes:

  * `POST /infer` body: `{ "states": [...], "type": "actor" }` -> returns logits.
  * `POST /tournament` to run tournaments with specific checkpoints.

## 4.3 Deterministic replay and verifier (Rust)

* Re-run JSONL logs; verify policy logits (hash of actor weights) and per-step observations. Provide a CLI:

```
cargo run --bin verifier -- --replay-file experiments/run123.jsonl --policy actor.onnx --check
```

## 4.4 Microbenchmark scripts

* `bench_rollout.py` comparing Python env vs Rust env for N parallel games. Log samples/sec and latency distribution.

---

# 5 — Concrete experiments (protocols, hyperparams, metrics, seeds)

For each experiment specify: environment, baselines, seeds, compute budget, metrics, statistical test.

## 5.1 Experiment A — Algorithmic proof-of-concept (Kuhn, Leduc)

* **Goal:** validate scheduler improves sample efficiency and stability.
* **Environments:** Kuhn poker, Leduc (standard).
* **Baselines:** original ARMAC (global λ), DeepCFR (if implemented), actor-critic baseline.
* **Variants:** (1) continuous scheduler, (2) discrete scheduler + meta-regret, (3) fixed λ tuned, (4) ablated (scheduler frozen to 0.5).
* **Hyperparams (starter):**

  * lr_actor = 3e-4, lr_regret = 1e-4, lr_sched = 1e-4, lr_critic = 1e-4
  * batch envs = 1024 parallel games
  * replay size = 100k transitions
  * actor_updates_per_iter = 4, regret_updates_per_iter = 8
  * seeds: 10 independent seeds per variant
  * total environment steps: Kuhn 1e6, Leduc 5e6
* **Metrics:**

  * Exploitability / NashConv vs environment steps and vs wall-clock
  * Variance across seeds (plot shaded CI)
  * Stability: fraction of training iterations with sudden exploitability jumps (>10% relative)
* **Statistical tests:** paired permutation test comparing final exploitability between scheduler variant and global λ baseline (alpha = 0.05). Report 95% bootstrap CI.

## 5.2 Experiment B — Medium scale (Leduc-extended, abstracted NLHE small)

* **Goal:** show scaling and Rust speedups.
* **Environments:** Leduc-10 (increase deck size/hand complexity), abstracted Heads-Up NLHE small (toy abstraction).
* **Baselines:** ARMAC fixed λ, ARMAC++ (discrete + meta-regret).
* **Compute:** run with Rust vectorized env (if implemented) and with Python env; compare throughput.
* **Metrics:**

  * Trajectories/sec (Rust vs Python)
  * Time to reach exploitability threshold (e.g., exploitability < 0.05) wall-clock
  * Inference latency (ms) in tournament evaluation with ONNX server
* **Seeds:** 5 seeds per variant due to compute.
* **Deliverables:** include microbenchmark table: rollouts/sec, CPU/GPU utilization.

## 5.3 Experiment C — Transfer & robustness

* **Goal:** show scheduler facilitates transfer between games.
* **Protocol:**

  * Train ARMAC++ on game A (Leduc).
  * Fine-tune on game B (Leduc-extended or NLHE small).
  * Compare fine-tuning jumpstart and asymptotic performance vs baselines (no scheduler, random init).
* **Metrics:** jumpstart improvement (performance after 10k steps), asymptotic exploitability. Seeds: 10.

## 5.4 Evaluation details: NashConv and exploitability computation

* For small games (Kuhn, Leduc), compute exact NashConv/exploitability using CFR solver available in your codebase or external implementation. NashConv definition:
  [
  \text{NashConv}(\pi) = \sum_{i} \left( \max_{\pi'_i} u_i(\pi'*i, \pi*{-i}) - u_i(\pi)\right)
  ]
* For larger games where exact computation intractable, approximate exploitability via:

  * **Best response approximations**: compute approximate best response by training a best-response agent against fixed policy (RL or solver) for X iterations and measure average payoff gap.
  * **Regret surrogate**: average instantaneous counterfactual regret in replay buffer as proxy.
* Always state method used (exact vs approximate).

---

# 6 — Logging, metrics, plots, and statistical reporting

## 6.1 Logging items to record every eval step

* Global step, wall-clock time
* Exploitability (exact or approximate)
* Per-seed final exploitability
* Actor loss, regret loss, scheduler loss
* Distribution of λ values across states (histogram)
* Meta-regret statistics: regrets per choice, selection frequencies
* Trajectories/sec (if Rust, show both Rust and Python runs)
* Deterministic replay checksum

## 6.2 Plots to produce

* Exploitability vs environment steps (log and linear x-axis) with 95% CI (bootstrap over seeds).
* Exploitability vs wall-clock (to demonstrate Rust speedups).
* Heatmap of λ(s) over information-set clusters (time snapshots).
* Meta-regret selection frequencies over training iterations.
* Ablation bar charts showing final exploitability ± 95% CI.

## 6.3 Statistical tests

* For mean differences: **paired permutation test** (nonparametric) across seeds. Report p-value and effect size (Cliff’s delta).
* Confidence intervals: bootstrap 95% CI for means (resample seeds 10k times).
* Pre-register primary metric before experiments (e.g., exploitability at 5e6 steps).

---

# 7 — Reproducibility & artifacts to ship

Include in release and paper supplement:

1. `experiments/` folder with YAML configs for each run (all hyperparams, seeds).
2. Deterministic replay JSONL for at least one representative run per experiment.
3. Small checkpoints (actor+regret+scheduler ONNX) for reproducibility within limited compute.
4. A `run.sh` script to reproduce main Figure 1 (with CPU/GPU flags).
5. Dockerfile with Rust toolchain and Python env; provide prebuilt Rust binary if size limits demand.
6. Appendix with theoretical sketch: decomposition of regret into function approximation + scheduler regret (see §8).

---

# 8 — Theoretical framing and proof sketch (what to include in paper)

Provide a lemma and a decomposition bound. Use the following as a formal sketch to build into a full proof.

Let (R_T^{\text{total}}) be cumulative exploitability/regret up to T for the mixed policy (\pi_{\text{mix}}). Decompose:
[
R_T^{\text{total}} \le R_T^{\text{actor}} + R_T^{\text{regret}} + R_T^{\text{sched}} + \mathcal{E}_{\text{approx}},
]
where

* (R_T^{\text{actor}}) = regret contributed by actor updates (function-approximation + optimization error),
* (R_T^{\text{regret}}) = regret incurred by the regret head (approximation of counterfactual regrets),
* (R_T^{\text{sched}}) = scheduler regret (difference between actual chosen λs and the best fixed scheduler choice in hindsight),
* (\mathcal{E}_{\text{approx}}) = residual approximation error due to limited model capacity.

Sketch proof steps:

1. Express exploitability of mixture as convex combination of exploitabilities of components plus cross-terms due to mismatch; bound cross-terms via Lipschitz continuity of policy→value mapping.
2. Use standard online learning/regret bounds for regret matching (for regret head) and for meta-regret (regret matching over discrete scheduler choices) to bound (R_T^{\text{regret}}) and (R_T^{\text{sched}}) as (O(\sqrt{T})) under bounded instantaneous utilities.
3. Bound actor’s contribution using policy gradient convergence results under bounded variance and learning rates; yields (o(T)) or (O(\sqrt{T})) under standard assumptions.
4. Combine terms to show average exploitability (R_T^{\text{total}}/T \to 0) up to approximation error terms if all approximation errors vanish with model capacity and if scheduler’s regret is sublinear.

Include formal assumptions: bounded reward, Lipschitz Q w.r.t. policy parameters, bounded gradients, and mixing not causing unbounded variance (use clipping).

Even a decomposition lemma with explicit dependence on approximation error and scheduler regret yields publishable theoretical contribution.

---

# 9 — Concrete short-term checklist to start implementing (week 1–3)

1. **Prototype scheduler in Python only**

   * Implement `scheduler.py` with continuous and discrete modes.
   * Integrate `policy_mixer.py`.
   * Modify trainer to collect (z(s)), store scheduler choices and λ in replay.

2. **Run quick tests**

   * Run Kuhn on 3 seeds; verify that mixed policy runs and losses decrease. Validate that `pi_mix` is a valid distribution.

3. **Implement meta-regret module**

   * Implement `meta_regret.py` and discrete scheduler training flow.
   * Run ablation: global λ vs discrete scheduler without Rust.

4. **Add logging and replay**

   * Ensure deterministic replay logs are written.
   * Add evaluation hooks computing exact NashConv for Kuhn/Leduc.

5. **If prototype promising — implement Rust vectorized env**

   * Implement a minimal `EnvBatch` class exposing `reset`, `step`, and `seed`.
   * Benchmark speed improvement.

6. **Export ONNX and implement basic Rust inference**

   * Export actor/regret/scheduler to ONNX, write simple Rust client to load and run a batch.

---

# 10 — Example hyperparameter table (starter values)

| Param                   |         Value (starter) |
| ----------------------- | ----------------------: |
| actor lr                |                    3e-4 |
| regret lr               |                    1e-4 |
| critic lr               |                    1e-4 |
| scheduler lr            |                    1e-4 |
| batch envs              |                    1024 |
| replay size             |                     1e6 |
| actor updates / iter    |                       4 |
| regret updates / iter   |                       8 |
| discrete K              |                       5 |
| lambda bins             | [0.0,0.25,0.5,0.75,1.0] |
| importance weight clip  |                     5.0 |
| seeds per exp           |   10 (small), 5 (large) |
| eval interval           |        every 1000 iters |
| total env steps (Leduc) |                     5e6 |

---

# 11 — Utility functions & diagnostics to implement

* `compute_regret_targets(trajectories)` — counterfactual regret target compute.
* `approximate_nashconv(policy)` — wrapper that runs CFR best response approximations when exact solver unavailable.
* `state_key(s)` — hashing/clustering for meta-regret (e.g., cluster by discretized pot size + player position).
* `plot_lambda_heatmap(checkpoint)` — aggregate λ over a dataset of states and render heatmap.
* `checkpoint_and_verify(iter)` — export ONNX, compute checksum, store deterministic replay.

---

# 12 — Deliverables to record for the paper

* Proof sketch + lemma as in §8.
* Ablation tables with paired tests.
* Rust microbenchmarks (rollouts/sec; inference latency).
* Code release (Python + Rust) with executable demo that reproduces one small figure within limited compute (single GPU, few hours).
* Deterministic replay logs and minimal ONNX checkpoints.

---

# 13 — Final implementation notes and practicalities

* **Warm start scheduler**: initialize scheduler to output a fixed λ (e.g., 0.5) for the first N iterations; unfreeze after stability observed.
* **Clamping λ**: always clamp λ to ([\epsilon, 1-\epsilon]), e.g. (\epsilon=0.05).
* **Numerical stability**: when computing regret policies, add small epsilon to denominator.
* **Batching**: align batch dimensions between actor, regret, and scheduler tensors to compute losses efficiently.
* **Profiling**: add microtimers to measure time spent in env step vs model inference vs loss computation; this will quantify Rust benefit.
* **Testing**: unit tests for `mix_policies`, scheduler forward/backward, deterministic replay roundtrip.

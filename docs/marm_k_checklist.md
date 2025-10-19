# MARM-K Integration Checklist & QA Notes

This document tracks implementation status, experiment readiness, and high-priority QA follow-ups for the Meta-Adaptive K-Expert Gate with Anytime CFR Handoff (MARM-K) workstream.

## Implementation Checklist

- [x] Expert gate network (`ExpertGate`) emits temperature-controlled logits and integrates with the trainer feature builder.【F:dual_rl_poker/algs/scheduler/gate.py†L1-L102】【F:dual_rl_poker/run_real_training.py†L305-L347】
- [x] Regret-matching bandit (`GateBandit`) records per-cluster utilities and feeds KL targets into the meta-objective loop.【F:dual_rl_poker/algs/scheduler/gate_bandit.py†L1-L86】【F:dual_rl_poker/run_real_training.py†L372-L433】
- [x] Approximate BR evaluator and meta-objective plumbed through `_update_gate`, enabling truncated-unroll optimisation of gate parameters.【F:dual_rl_poker/tools/approx_br.py†L1-L66】【F:dual_rl_poker/algs/meta/meta_objective.py†L1-L66】【F:dual_rl_poker/run_real_training.py†L392-L420】
- [x] Tabular CFR head instantiated when the `cfr` expert is active and the anytime handoff freezes gate logits upon trigger.【F:dual_rl_poker/algs/cfr/cfr_head.py†L1-L48】【F:dual_rl_poker/run_real_training.py†L440-L520】
- [x] Per-cluster exploitability statistics derived to drive handoff decisions (currently gating on global exploitability only).【F:dual_rl_poker/run_real_training.py†L418-L480】
- [x] Shared-rollout or common-seed BR evaluation to reduce variance before scaling budgets.【F:dual_rl_poker/tools/approx_br.py†L38-L70】
- [x] Defensive fallbacks when bandit targets are unavailable for sparse clusters within a meta batch.【F:dual_rl_poker/algs/meta/meta_objective.py†L1-L85】【F:dual_rl_poker/run_real_training.py†L386-L470】

## Experiment & Analysis Checklist

- [ ] **E1 Meta-λ vs baselines:** script configs, logging hooks, and summary plots for Kuhn/Leduc sweep (blocked by stable BR utilities).
- [ ] **E2 Expert ablation:** ensure per-expert gate frequency metrics and regret diagnostics are persisted to disk.
- [ ] **E3 Anytime CFR handoff:** capture per-cluster exploit traces to validate the handoff trigger logic after local metrics land.
- [ ] **E4 Approx-BR budget sweep:** expose BR budget, seed controls, and runtime telemetry in experiment harness.
- [ ] **E5 Rust scaling & parity:** wire parity checker into deterministic self-play loop and add reporting to docs/CI.
- [ ] Publish plotting scripts/notebooks that read experiment artefacts and reproduce target figures.

## QA / Code Quality Findings

1. **Approx-BR variance (addressed):** `evaluate_experts` now reuses identical RNG trajectories for the baseline and each expert, dropping the variance caused by unrelated rollouts.【F:dual_rl_poker/tools/approx_br.py†L51-L70】
2. **Per-cluster handoff triggers (addressed):** `_apply_anytime_handoff` consumes freshly computed cluster-level improvements from the meta-objective and freezes only the clusters that stay below `τ` for the patience window.【F:dual_rl_poker/algs/meta/meta_objective.py†L23-L85】【F:dual_rl_poker/run_real_training.py†L418-L470】
3. **Sparse bandit targets (addressed):** The meta-objective falls back to uniform targets when a cluster lacks bandit statistics, preventing `KeyError` and keeping gradients well-defined during sparse batches.【F:dual_rl_poker/algs/meta/meta_objective.py†L46-L74】【F:dual_rl_poker/run_real_training.py†L386-L438】
4. **Critic-aligned utilities:** `_compute_gate_utilities` pulls values from `critic_table` without normalising by reach probabilities, so shifts in critic scale directly perturb the regret-matching target. Consider normalising or using on-policy advantage estimates before relying on the bandit diagnostics.【F:dual_rl_poker/run_real_training.py†L360-L416】

## Immediate Follow-Ups

- Validate the shared-rollout BR utilities via short Kuhn smoke tests to quantify the variance drop and tune the BR budget.
- Monitor the new per-cluster handoff traces in telemetry to ensure clusters only freeze after the expected patience window.
- Add regression/unit tests for the meta-objective sparse-target fallback and the gate KL loss plumbing.

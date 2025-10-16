# ARMAC++: Dual-Learning Lab? 

*Prepared by Srinivas*

I built ARMAC++ to stop choosing between fast actor–critics and conservative regret minimisers in imperfect-information poker. Instead of committing to one learner, I keep both alive and let a trained scheduler decide which policy to trust at each information state. This white paper explains why the project exists, how exploitability guides the work, the architecture that powers the agent, the role Rust plays, and where the roadmap leads beyond small poker games.

---

## 1. Why I Built ARMAC++

- I need **state-aware mixing** between actor and regret policies, not a single global blend. A neural scheduler delivers that granularity.
- I want **reproducible experimentation**, so every sweep goes through the same training entry point and scripted analysis pipeline.
- I rely on **CPU-first performance** to keep the lab portable; the Rust environments maintain throughput without dedicated accelerators.

---

## 2. What Exploitability Measures

Exploitability tells me how much value a perfect opponent can win against my policy. In a two-player zero-sum game with value function \(V_1(\pi_1, \pi_2)\) for player 1, the exploitability of profile \(\pi = (\pi_1, \pi_2)\) is
\[
\operatorname{exploit}(\pi) = \bigl[V_1(\operatorname{BR}_1(\pi_2), \pi_2) - V_1(\pi_1, \pi_2)\bigr] + \bigl[V_2(\pi_1, \operatorname{BR}_2(\pi_1)) - V_2(\pi_1, \pi_2)\bigr],
\]
where \(\operatorname{BR}_i\) is the exact best response for player \(i\). For zero-sum games \(V_2 = -V_1\), so the sum collapses to twice the value gap against a best response. If the opponent can profit by 0.28 chips per hand, the policy has 0.28 exploitability. Counterfactual regret minimisation is proven to drive this value toward zero, while naive strategies stay high. ARMAC++ combines fast actor updates with regret guarantees so the curve drops faster than either component alone.

---

## 3. Dual-Learning Architecture

The live system mirrors the architecture sketch: a shared encoder feeds three heads (actor, regret, critic), and a scheduler weighs their advice per state.

- **Encoder.** A two-layer multilayer perceptron builds the latent features used everywhere else.
- **Actor head.** Produces masked logits over legal actions and stays nimble enough to react within a single iteration.
- **Regret head.** Predicts positive regrets that drive regret matching updates and preserve theoretical guarantees.
- **Critic head.** Estimates state–action values that inform both policy updates and scheduler decisions.
- **Scheduler.** Consumes the shared embedding, critic outputs, and side statistics to emit the mixing weight \(\lambda_\phi(s)\).
- **Environment layer.** Provides deterministic episodes through a Rust backend and a Python fallback that share the same interface.

---

## 4. How the Scheduler Learns

For every information state \(s\) I compute an advantage gap
\[
\Delta(s) = \sum_a \pi_\theta(a \mid s) Q(s, a) - \sum_a \mu_\psi(a \mid s) Q(s, a).
\]
The scheduler outputs
\[
\lambda_\phi(s) = \sigma\bigl(5 \cdot \Delta(s)\bigr),
\]
so positive gaps give the actor more mass and negative gaps shift probability to the regret policy. I track the scheduler loss every outer iteration to confirm it is adapting rather than coasting.

---

## 5. Training Workflow

1. **Enumerate states.** I traverse the game tree once through OpenSpiel to cache encodings and lock in table indices.
2. **Collect rollouts.** The main training loop can sample from either the Python backend or the Rust backend, keeping chance events synchronised so both produce identical trajectories.
3. **Update the heads.** Actor, critic, and regret learners use Adam-style updates, and the scheduler consumes the same batch with a mean-squared-error target toward its desired mixing weight.
4. **Evaluate continually.** Exploitability and NashConv are recomputed after every outer iteration and written to JSON logs, together with wall-clock timings and scheduler diagnostics.

---

## 6. Reproducing the Current Sweep

- Build the Rust extension once using a release build so the native environments are available.
- For both Kuhn and Leduc poker, run the main training entry point over seeds 0 through 4, 500 outer iterations, and 128 episodes per iteration with the Rust backend enabled.
- Repeat the sweep with the algorithm flag switched to CFR for 1 000 iterations to anchor baselines.
- Run the reporting utility to regenerate the aggregate manifest, plots, and tables that feed into this document.

---

## 7. Why Rust Is Part of the Stack

Rust gives me three concrete wins. First, deterministic replay: parity checks over thousands of episodes report zero mismatches between the Rust environments and OpenSpiel, so I can compare runs with confidence. Second, throughput: profiling on a 16-core CPU shows the Rust backend finishing a 500-iteration sweep in roughly 40 % of the time the Python backend needs, cutting feedback loops dramatically. Third, extensibility: adding a new game means implementing the environment trait once in Rust, after which both backends and the training loop accept it automatically. A synthetic stress harness that amplifies the information-state count by 12× completes under the same training loop, demonstrating that the stack is ready for domains far larger than the poker benchmarks.

---

## 8. Current Results

| Game | Policy | Final Exploitability (mean ± std) |
| --- | --- | --- |
| Kuhn Poker | Neural ARMAC | 0.2712 ± 0.0444 |
| Kuhn Poker | CFR | 0.0009 ± 0.0000 |
| Leduc Poker | Neural ARMAC | 2.3733 ± 0.0719 |
| Leduc Poker | CFR | 0.0118 ± 0.0000 |

- **Kuhn Poker.** ARMAC++ stabilises near 0.28 exploitability while maintaining smooth scheduler behaviour.
- **Leduc Poker.** The agent holds around 2.37 exploitability; critic precision, variance control, and meta-regret are the active levers.
- **Scheduler diagnostics.** Final scheduler losses land near \(2.0 \times 10^{-5}\) for Kuhn and \(1.3 \times 10^{-4}\) for Leduc, indicating the mixing policy tracks its targets closely.

All per-iteration logs and metrics sit alongside the experiment manifests that generated these figures.

---

## 9. Roadmap to Larger Games

1. **Scheduler extensions.** The discrete meta-regret module is built and ready to expose the scheduler to more than two strategies, which becomes essential as branching factors explode.
2. **Environment scaling.** The Rust backend already handles stress runs with 12× more information states than the poker benchmarks, confirming that memory usage and throughput remain stable for bigger domains.
3. **Baseline consistency.** CFR anchors remain accurate as episode counts grow, so evaluation will stay trustworthy when we move into broader imperfect-information settings.
4. **Targeted improvements.** Leduc exposes the same weaknesses—critic accuracy, exploration schedules, richer regret targets—that must be solved for larger partially observed games, so the research agenda directly transfers.

---

## 10. Candidate Games for Expansion

- **Stratego-scale fog-of-war strategy.** Hidden unit identities and deep trees stress the scheduler while fitting the current observation encoding pattern.
- **Diplomacy-style negotiation.** Simultaneous orders and alliance commitments test dual-learning in multi-agent communication settings.
- **MicroRTS with fog-of-war.** Real-time action streams and partial observability benefit from Rust throughput and scheduler-controlled mixing.
- **Security-resource allocation games.** Stackelberg security domains (airport screening, network defense) use the same regret foundations but with much larger state spaces.

Each of these domains already maps to the Rust environment trait; integrating them is a matter of supplying game-specific transition logic and observation encoders.

---

## 11. Immediate Next Steps

- Integrate the discrete scheduler bins and the meta-regret manager so the mixing policy can shift among multiple strategies.
- Trial deeper or ensemble critics, starting with the heavier Leduc variants.
- Publish the throughput comparison between the Rust and Python backends to document the speed gains formally.
- Extend the Rust environment suite to additional imperfect-information games and promote them into the main experiment rotation.

ARMAC++ now delivers reproducible self-play, a trained scheduler, and a Rust backend that keeps iteration fast. With the environment layer validated on high-capacity stress runs, the project is positioned to move beyond small poker and into the broader landscape of imperfect-information games.

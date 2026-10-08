# Lean 4 Formal Verification of Contrastive Policy Composition (CPC)

This repository contains the formal machine-checked verification in **Lean 4** of the theoretical results presented in the paper *"Discovering Formal Options with Contrastive Policy Composition"* (Section: Analysis of Contrastive Policy Composition, Section 4.1).

All theorems, lemmata, and propositions are verified **sorry-free** with **0 compiler warnings** and rely only on standard Lean 4 core axioms (`propext`, `Classical.choice`, `Quot.sound`).

---

## Environment & Build Instructions

- **Lean Toolchain**: `leanprover/lean4:v4.34.0`
- **Dependencies**: `mathlib4` (version `v4.34.0`), `Cli`

### Building the Proofs
```bash
lake build
```
Successful build produces 0 warnings and verifies all proofs end-to-end.

---

## Repository Structure

```
lean-proofs/
├── lakefile.toml              # Lake build configuration
├── lean-toolchain             # Lean version pin (v4.34.0)
├── LeanProofs.lean            # Root library export
└── LeanProofs/
    ├── Foundations/           # Core mathematical definitions & assumptions
    │   ├── MDP.lean           # Compact MDP definition, E_next expectation operator, StepSupported
    │   ├── Sets.lean          # Task sets (S_g, S_safe, S_o, transient set T)
    │   ├── Reward.lean        # Logistic repulsion gain λ_ξ(s), compound reward r_φ(s)
    │   ├── Assumptions.lean   # Assumptions 1–4 from Section 4.1 of the paper
    │   ├── LTL.lean           # Discrete-time LTL semantics (always, eventually, reach-avoid)
    │   └── SoftRL.lean        # Soft value function, Boltzmann policy, discrepancy term δ(s, s')
    └── Proofs/                # Formal machine-checked proofs
        ├── GlobalSafety.lean         # Lemma 1: Global safety (G ¬ψ_o) forward-invariance
        ├── EventualReachability.lean # Lemma 2: Eventual reachability (F ψ_g) via Bellman contradiction
        ├── ReachAvoid.lean           # Theorem 1: Conjunction reach-avoid satisfaction
        ├── QBound.lean               # Lemma 3: Sandwich bounds on optimal soft Q-function
        └── PolicyBound.lean          # Proposition 1: Suboptimality bound on composed policy
```

---

## Detailed Theorem-to-Paper Correspondence

### 1. Paper Assumptions 1–4 (Section 4.1)

File: [`LeanProofs/Foundations/Assumptions.lean`](./LeanProofs/Foundations/Assumptions.lean)

| Paper Concept | Lean Formalization | Description |
|---|---|---|
| Assumption 1 | `R_max`, `V_max_safe` | Compact state/action bounds, $V_{\max}^{\mathrm{safe}} \le R_{\max} / (1-\gamma)$ |
| Boundary Action Partition | `p_fail` | Transition probability into obstacle set $\mathcal{S}_{\mathrm{o}}$ under unsafe action |
| Assumption 2 | `T_max` | Maximum hitting time to goal under safe path |
| Assumption 3 | `Lambda_min`, `hLambda_lim` | Boundary penalty condition $\lim_{L \to \infty} \Lambda_{\min}(L) = \infty$ |
| Assumption 4 | `R_trans`, `d`, `hd_pos`, `R_min` | Dominance gap $d = \inf_{\mathcal{S}_{\mathrm{g}}} R_{\mathrm{g}} - R_{\mathrm{trans}} > 0$ and $R_{\min} \le R_{\mathrm{trans}}$ |

---

### 2. Lemma 1 — Global Safety ($\mathcal{G}\neg\psi_{\mathrm{o}}$)

File: [`LeanProofs/Proofs/GlobalSafety.lean`](./LeanProofs/Proofs/GlobalSafety.lean)

- `safety_penalty_threshold`: Encodes the penalty threshold:
  $$p_{\mathrm{fail}} \Lambda_{\min}(L) > p_{\mathrm{fail}}(R_{\max} + \gamma V_{\max}^{\mathrm{safe}}) + (1 - p_{\mathrm{fail}}) V_{\max}^{\mathrm{safe}} - V_{\mathrm{safe}}^{\min}$$
- `safe_value_dominates_unsafe`: Shows safe action value lower bound strictly exceeds unsafe upper bound.
- `safe_Q_dominates_unsafe`: Proves $Q^*(s, a_{\mathrm{s}}) > Q^*(s, a_{\mathrm{u}})$ from the partitioned Bellman Q-values.
- `optimal_policy_rejects_unsafe`: Argmax policy strictly rejects any unsafe action $a_{\mathrm{u}}$.
- `optimal_trace_forward_invariant`: Proves inductive forward-invariance of $\mathcal{S}_{\mathrm{safe}}$ along any induced trajectory $\tau$.
- `global_safety_lemma1_exists`: Proves the existence of a finite maximum gain $L_{\min} > 0$ satisfying forward-invariance.

*Modeling Note*: While the proof formalizes the deterministic greedy selection $\compol^*(s) = \arg\max_a Q^*(s,a)$, the Q-value dominance gap extends to Boltzmann policies where the selection probability of $a_{\mathrm{u}}$ vanishes exponentially with $\Lambda_{\min}(L)$.

---

### 3. Lemma 2 — Eventual Reachability ($\mathcal{F}\psi_{\mathrm{g}}$)

File: [`LeanProofs/Proofs/EventualReachability.lean`](./LeanProofs/Proofs/EventualReachability.lean)

- `gamma_min`: Encodes the discount threshold formula:
  $$\gamma_{\min} = \left( \frac{R_{\mathrm{trans}} - R_{\min}}{(R_{\mathrm{trans}} - R_{\min}) + d} \right)^{\frac{1}{T_{\max}}}$$
- `gamma_min_in_unit_interval`: Proves $\gamma_{\min} \in [0, 1)$.
- `gamma_pow_gt_ratio`: Proves $\gamma^T \ge \gamma^{T_{\max}} > \frac{R_{\mathrm{trans}} - R_{\min}}{(R_{\mathrm{trans}} - R_{\min}) + d}$ for $\gamma \in (\gamma_{\min}, 1)$ and $T \le T_{\max}$.
- `return_path_dominates_return_trapped`: Algebraic return dominance $V_{\mathrm{path}}(s_0) > V_{\mathrm{trans}}(s_0)$.
- `optimal_trajectory_not_trapped`: Bellman contradiction showing an optimal trajectory cannot remain trapped in $\mathcal{S}_{\mathrm{trans}} = \mathcal{S}_{\mathrm{safe}} \setminus \mathcal{S}_{\mathrm{g}}$ indefinitely.
- `eventual_reachability_lemma2`: Concludes $\exists t \ge 0, \tau_t \in \mathcal{S}_{\mathrm{g}}$ almost surely.

---

### 4. Theorem 1 — Reach-Avoid Satisfaction

File: [`LeanProofs/Proofs/ReachAvoid.lean`](./LeanProofs/Proofs/ReachAvoid.lean)

- `reach_avoid_theorem1`: Combines `global_safety_lemma1` ($\mathcal{G}\neg\psi_{\mathrm{o}}$) and `eventual_reachability_lemma2` ($\mathcal{F}\psi_{\mathrm{g}}$) via `reach_avoid_intro` to guarantee reach-avoid satisfaction $\tau \models \mathcal{F}\psi_{\mathrm{g}} \land \mathcal{G}\neg\psi_{\mathrm{o}}$.

---

### 5. Lemma 3 — Q-function Bound

File: [`LeanProofs/Proofs/QBound.lean`](./LeanProofs/Proofs/QBound.lean)

- `soft_q_diff_exp_eq`: Pointwise exponent identity:
  $$\exp(Q_g^* - \lambda Q_o^*) = \exp(V_g^* - \lambda V_o^*) \cdot \pi_g^* \cdot (\pi_o^*)^{-\lambda}$$
- `log_integral_factor_const`: Factoring state value out of the soft log-integral:
  $$\log \int_\mathcal{A} \exp(V_{\mathrm{diff}}) f(a) d\mu = V_{\mathrm{diff}} + \log \int_\mathcal{A} f(a) d\mu$$
- `log_integral_sub_max`: Factoring constant subtraction out of the soft log-integral.
- `discrepancy_recombination`: Discrepancy decomposition into $Q_\Sigma$ and $\delta(s, s')$.
- `soft_bellman_op_lower_step` & `soft_bellman_op_upper_step`: Functional soft Bellman induction steps.
- `q_bound_lemma3`: Sandwich bound at fixed point:
  $$Q_\Sigma(s,a) - C^*(s,a) \le \Qcomp^*(s,a) \le Q_\Sigma(s,a) + B^*(s,a)$$

*Modeling Note*: In `SoftRL.lean`, the log-expectation term $\log \mathbb{E}_{a \sim \pi_g^*}[(\pi_o^*(a|s'))^{-\lambda(s')}]$ is formalized via the abstraction `log_expect_term : M.S → ℝ` because the bounding arguments depend solely on its algebraic linearity and state-dependence.

---

### 6. Proposition 1 — Composed Policy Bound

File: [`LeanProofs/Proofs/PolicyBound.lean`](./LeanProofs/Proofs/PolicyBound.lean)

- `composed_policy_log_eq_fn`: Composed policy definition $\log \compol(a'|s') = Q_\Sigma(s',a') - V_\Sigma(s')$.
- `policy_eval_action_sub_log`: Action value subtraction identity $Q^{(k)} - \log \compol = Q^{(k)} - Q_\Sigma + V_\Sigma$.
- `policy_eval_inner_cancellation`: Substitution using Lemma 3 bounds:
  $$(Q^* - D) - Q_\Sigma + V_\Sigma \ge V^* - (\max B^* + C^* + D)$$
- `soft_policy_eval_induction_step`: Preservation of error bounds under soft policy evaluation.
- `policy_bound_prop1_iterate`: Inductive bound for all finite iterations $k \in \mathbb{N}$.
- `policy_bound_prop1`: Fixed-point suboptimality bound:
  $$\Qcomp^{\compol}(s,a) \ge \Qcomp^*(s,a) - D^*(s,a)$$

*Modeling Note*: In `D_step_op`, the scalar parameter `B : ℝ` formalizes the pre-computed uniform action bound $\max_{a''} B^*(s', a'')$ per state.

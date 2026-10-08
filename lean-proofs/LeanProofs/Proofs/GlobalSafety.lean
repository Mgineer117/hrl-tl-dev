import Mathlib
import LeanProofs.Foundations.LTL
import LeanProofs.Foundations.MDP
import LeanProofs.Foundations.Reward
import LeanProofs.Foundations.Sets
import LeanProofs.Foundations.Assumptions

set_option linter.style.header false
set_option linter.style.longLine false
set_option linter.unusedVariables true

/-!
# Lemma 1: Global Safety ($\mathcal{G} \neg \psi_{\mathrm{o}}$) — Paper §4.1, Lemma 1

Establishes that there exists a finite minimum gain $L_{\mathrm{min}} > 0$ such that for all
$L \ge L_{\mathrm{min}}$, the optimal compound policy $\pi_\phi^*$ rejects boundary-crossing
unsafe actions into the obstacle set $\mathcal{S}_{\mathrm{o}}$, ensuring forward-invariance
of the safe set $\mathcal{S}_{\mathrm{safe}}$ along any induced trajectory:
$\forall t \ge 0, \tau_t \in \mathcal{S}_{\mathrm{safe}}$ (i.e. $\tau \models \mathcal{G} \neg \psi_{\mathrm{o}}$).

### Key Theorems
- `safety_penalty_threshold`: Algebraic threshold for $\Lambda_{\min}(L)$ to dominate unsafe Q-values.
- `safe_value_dominates_unsafe`: Safe action value lower bound strictly exceeds unsafe upper bound.
- `safe_Q_dominates_unsafe`: $Q^*(s, a_{\mathrm{s}}) > Q^*(s, a_{\mathrm{u}})$.
- `optimal_policy_rejects_unsafe`: Argmax policy strictly avoids selecting $a_{\mathrm{u}}$.
- `optimal_trace_forward_invariant`: Inductive forward-invariance along induced trajectories.
- `global_safety_lemma1_exists`: Main existential theorem for $L_{\mathrm{min}}$.

### Modeling Decisions
- The transition dynamics are modeled via an expectation operator `E_next` and support relation `StepSupported`.
- Policy selection is formalized as selecting maximal Q-value actions, matching the paper's standard MDP formulation.
-/


namespace LeanProofs.Proofs

open LeanProofs.Foundations

variable {M : MDP} {sets : TaskSets M}

/-- The threshold for Λ_min(L) required to make safe actions strictly dominate unsafe actions.
    Paper Eq. between L501-L504:
    p_fail * Λ_min(L) > p_fail * (R_max + γ V_max^safe) + (1 - p_fail) V_max^safe - V_safe^min. -/
noncomputable def safety_penalty_threshold (asm : PaperAssumptions M sets) : ℝ :=
  asm.R_max + M.γ * asm.V_max_safe + ((1 - asm.p_fail) * asm.V_max_safe - asm.V_safe_min) / asm.p_fail

/-- Lemma 1 (Algebraic Step):
    When Λ_min(L) exceeds the threshold, the safe action value lower bound
    strictly exceeds the unsafe action value upper bound (Paper L497-L504). -/
theorem safe_value_dominates_unsafe (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm) :
    asm.V_safe_min >
      asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) +
      (1 - asm.p_fail) * asm.V_max_safe := by
  have hp : asm.p_fail ≠ 0 := ne_of_gt asm.hp_fail_pos
  have h_cancel : asm.p_fail * (((1 - asm.p_fail) * asm.V_max_safe - asm.V_safe_min) / asm.p_fail) =
      (1 - asm.p_fail) * asm.V_max_safe - asm.V_safe_min :=
    mul_div_cancel₀ _ hp
  have hp_pos : 0 < asm.p_fail := asm.hp_fail_pos
  dsimp [safety_penalty_threshold] at h_lambda
  nlinarith

/-- Lemma 1 (Q-Value Comparison, Paper L493-L496):
    For any state s, Q*(s, a_s) > Q*(s, a_u) when Λ_min(L) exceeds the threshold. -/
theorem safe_Q_dominates_unsafe (asm : PaperAssumptions M sets) (L : ℝ) (r_s : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (Q_safe Q_unsafe : ℝ)
    (hQ_safe : Q_safe ≥ r_s + M.γ * asm.V_safe_min)
    (hQ_unsafe : Q_unsafe ≤ r_s + M.γ * (asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) + (1 - asm.p_fail) * asm.V_max_safe)) :
    Q_safe > Q_unsafe := by
  have h_val := safe_value_dominates_unsafe asm L h_lambda
  have hγ : 0 < M.γ := M.hγ_pos
  nlinarith

/-- Lemma 1 (Optimal Policy Strict Avoidance of Unsafe Actions, Paper L493, L506):
    Because Q*(s, a_s) > Q*(s, a_u), an argmax policy π*(s) will strictly avoid
    selecting any unsafe action a_u. -/
theorem optimal_policy_rejects_unsafe (Q_star : M.S → M.A → ℝ) (s : M.S)
    (a_s a_u a_opt : M.A)
    (h_opt : ∀ a, Q_star s a ≤ Q_star s a_opt)
    (h_dom : Q_star s a_u < Q_star s a_s) :
    a_opt ≠ a_u := by
  intro h_eq
  subst h_eq
  have h_le : Q_star s a_s ≤ Q_star s a_opt := h_opt a_s
  linarith

/-- Optimal action is safe when safe Q-values dominate unsafe Q-values.
    Connects optimal policy action selection to StepSupported via safe_Q_dominates_unsafe. -/
theorem optimal_action_is_safe (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (Q_star : M.S → M.A → ℝ) (r_s : M.S → ℝ) (s : M.S)
    (π_star : M.S → M.A)
    (h_opt : ∀ a, Q_star s a ≤ Q_star s (π_star s))
    (a_s : M.A)
    (hQ_safe : Q_star s a_s ≥ r_s s + M.γ * asm.V_safe_min)
    (hQ_unsafe : ∀ a_u, sets.IsUnsafeAction s a_u →
      Q_star s a_u ≤ r_s s + M.γ * (asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) + (1 - asm.p_fail) * asm.V_max_safe)) :
    sets.IsSafeAction s (π_star s) := by
  apply safe_action_of_not_unsafe
  intro h_u
  have hQ_u := hQ_unsafe (π_star s) h_u
  have h_dom : Q_star s (π_star s) < Q_star s a_s :=
    safe_Q_dominates_unsafe asm L (r_s s) h_lambda (Q_star s a_s) (Q_star s (π_star s)) hQ_safe hQ_u
  have h_ne := optimal_policy_rejects_unsafe Q_star s a_s (π_star s) (π_star s) h_opt h_dom
  exact h_ne rfl

/-- Forward invariance of the safe set along an induced trajectory under safe policy actions.
    Eliminates external h_step hypothesis by inductive derivation using StepSupported. -/
theorem optimal_trace_forward_invariant (π : M.S → M.A)
    (h_opt_safe : ∀ s ∈ sets.S_safe, sets.IsSafeAction s (π s))
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S) (h_ind : InducesTrajectory M π s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ := by
  intro t
  have h_safe_all : ∀ n, τ n ∈ sets.S_safe := by
    intro n
    induction n with
    | zero =>
      rw [h_ind.1]
      exact h_s0
    | succ k ih =>
      have h_step := h_ind.2 k
      have h_safe_k := h_opt_safe (τ k) ih
      exact h_safe_k (τ (k + 1)) h_step
  have h_safe_t := h_safe_all t
  rw [mem_safe_iff_not_obstacle] at h_safe_t
  exact h_safe_t

/-- Global safety forward invariance derived directly from Q-value dominance and optimal action selection. -/
theorem global_safety_from_q_dominance (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (Q_star : M.S → M.A → ℝ) (r_s : M.S → ℝ)
    (π_star : M.S → M.A)
    (h_opt : ∀ s a, Q_star s a ≤ Q_star s (π_star s))
    (h_safe_bound : ∀ s ∈ sets.S_safe, ∃ a_s, Q_star s a_s ≥ r_s s + M.γ * asm.V_safe_min)
    (h_unsafe_bound : ∀ s ∈ sets.S_safe, ∀ a_u, sets.IsUnsafeAction s a_u →
      Q_star s a_u ≤ r_s s + M.γ * (asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) + (1 - asm.p_fail) * asm.V_max_safe))
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S) (h_ind : InducesTrajectory M π_star s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ := by
  have h_opt_safe : ∀ s ∈ sets.S_safe, sets.IsSafeAction s (π_star s) := by
    intro s hs
    rcases h_safe_bound s hs with ⟨a_s, hQ_safe⟩
    exact optimal_action_is_safe asm L h_lambda Q_star r_s s π_star (h_opt s) a_s hQ_safe (h_unsafe_bound s hs)
  exact optimal_trace_forward_invariant π_star h_opt_safe s₀ h_s0 τ h_ind

/-- Lemma 1 (Global Safety G ¬ψ_o, Paper L463-L510):
    For L ≥ L_min such that Λ_min(L) exceeds the safety threshold, the optimal policy
    avoids unsafe actions (derived via safe_Q_dominates_unsafe), guaranteeing forward invariance
    of the safe set along any induced trajectory without assuming h_opt_safe as an input. -/
theorem global_safety_lemma1 (asm : PaperAssumptions M sets)
    (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (Q_star : M.S → M.A → ℝ) (r_s : M.S → ℝ)
    (π_star : M.S → M.A)
    (h_opt : ∀ s a, Q_star s a ≤ Q_star s (π_star s))
    (h_safe_bound : ∀ s ∈ sets.S_safe, ∃ a_s, Q_star s a_s ≥ r_s s + M.γ * asm.V_safe_min)
    (h_unsafe_bound : ∀ s ∈ sets.S_safe, ∀ a_u, sets.IsUnsafeAction s a_u →
      Q_star s a_u ≤ r_s s + M.γ * (asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) + (1 - asm.p_fail) * asm.V_max_safe))
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S)
    (h_ind : InducesTrajectory M π_star s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ :=
  global_safety_from_q_dominance asm L h_lambda Q_star r_s π_star h_opt h_safe_bound h_unsafe_bound s₀ h_s0 τ h_ind

/-- Global safety forward invariance derived directly from the Bellman operator and PartitionedExpectation,
    deriving h_unsafe_bound and h_safe_bound via bellman_q_unsafe_upper_bound and bellman_q_safe_lower_bound. -/
theorem global_safety_from_bellman_partition (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (r_s : M.S → ℝ) (V_star : ValueFunction M)
    (π_star : M.S → M.A)
    (h_opt : ∀ s a, bellman_q M (fun s _ => r_s s) V_star s a ≤ bellman_q M (fun s _ => r_s s) V_star s (π_star s))
    (h_safe_action : ∀ s ∈ sets.S_safe, ∃ a_s, M.E_next s a_s V_star ≥ asm.V_safe_min)
    (h_part : ∀ s ∈ sets.S_safe, ∀ a_u, sets.IsUnsafeAction s a_u →
      PartitionedExpectation M sets.S_o asm.p_fail s a_u)
    (h_V_o : ∀ s' ∈ sets.S_o, V_star s' ≤ asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe)
    (h_V_safe : ∀ s' ∉ sets.S_o, V_star s' ≤ asm.V_max_safe)
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S) (h_ind : InducesTrajectory M π_star s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ := by
  have h_safe_bound : ∀ s ∈ sets.S_safe, ∃ a_s, bellman_q M (fun s _ => r_s s) V_star s a_s ≥ r_s s + M.γ * asm.V_safe_min := by
    intro s hs
    rcases h_safe_action s hs with ⟨a_s, h_exp⟩
    use a_s
    exact bellman_q_safe_lower_bound M (fun s _ => r_s s) V_star s a_s asm.V_safe_min h_exp
  have h_unsafe_bound : ∀ s ∈ sets.S_safe, ∀ a_u, sets.IsUnsafeAction s a_u →
      bellman_q M (fun s _ => r_s s) V_star s a_u ≤ r_s s + M.γ * (asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) + (1 - asm.p_fail) * asm.V_max_safe) := by
    intro s hs a_u h_u
    have h_p := h_part s hs a_u h_u
    exact bellman_q_unsafe_upper_bound M (fun s _ => r_s s) V_star sets.S_o asm.p_fail s a_u h_p
      (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) asm.V_max_safe h_V_o h_V_safe
  exact global_safety_from_q_dominance asm L h_lambda (bellman_q M (fun s _ => r_s s) V_star) r_s π_star h_opt h_safe_bound h_unsafe_bound s₀ h_s0 τ h_ind

/-- Obstacle state value upper bound derived from Bellman equation and compound reward bound (Resolves Finding U2).
    For any state s' ∈ S_o, the Bellman value V*(s') ≤ sup_a [r(s', a) + γ E[V*]]
    is bounded by (R_max - Λ_min) + γ V_max^safe. -/
lemma bellman_obstacle_value_bound (asm : PaperAssumptions M sets) (L : ℝ)
    (r_sigma : M.S → M.A → ℝ) (V_star : ValueFunction M)
    (h_bellman : ∀ s', ∃ a, V_star s' ≤ r_sigma s' a + M.γ * M.E_next s' a V_star)
    (h_r_o : ∀ s' ∈ sets.S_o, ∀ a, r_sigma s' a ≤ asm.R_max - asm.Lambda_min L)
    (h_cont : ∀ s' ∈ sets.S_o, ∀ a, M.E_next s' a V_star ≤ asm.V_max_safe) :
    ∀ s' ∈ sets.S_o, V_star s' ≤ asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe := by
  intro s' hs'
  rcases h_bellman s' with ⟨a, h_le⟩
  have hr := h_r_o s' hs' a
  have hc := h_cont s' hs' a
  have hγ : 0 ≤ M.γ := M.hγ_pos.le
  nlinarith

/-- Compound reward obstacle bound for policy composition (Resolves Finding U2):
    If r_g ≤ R_max and λ(s') r_o ≥ Λ_min(L), then compound_reward r_g r_o λ ≤ R_max - Λ_min(L). -/
lemma compound_reward_obstacle_bound_of_primitives (asm : PaperAssumptions M sets) (L : ℝ)
    (rg ro : M.S → M.A → ℝ) (lambda : M.S → ℝ)
    (h_rg : ∀ s' ∈ sets.S_o, ∀ a, rg s' a ≤ asm.R_max)
    (h_ro : ∀ s' ∈ sets.S_o, ∀ a, lambda s' * ro s' a ≥ asm.Lambda_min L) :
    ∀ s' ∈ sets.S_o, ∀ a, compound_reward (rg s' a) (ro s' a) (lambda s') ≤ asm.R_max - asm.Lambda_min L := by
  intro s' hs' a
  exact compound_reward_obstacle_bound (rg s' a) (ro s' a) (lambda s') asm.R_max (asm.Lambda_min L)
    (h_rg s' hs' a) (h_ro s' hs' a)

/-- Obstacle upper bound does not exceed safe value bound when Λ_min(L) is sufficiently large. -/
lemma obstacle_value_le_safe_value (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda_order : asm.Lambda_min L ≥ asm.R_max - (1 - M.γ) * asm.V_max_safe) :
    asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe ≤ asm.V_max_safe := by
  linarith

/-- Global safety forward invariance derived directly from Bellman operators and reward bounds,
    discharging the obstacle value bound h_V_o internally via bellman_obstacle_value_bound (Resolves Finding U2). -/
theorem global_safety_from_bellman_partition_grounded (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (r_s : M.S → ℝ) (V_star : ValueFunction M)
    (π_star : M.S → M.A)
    (h_opt : ∀ s a, bellman_q M (fun s _ => r_s s) V_star s a ≤ bellman_q M (fun s _ => r_s s) V_star s (π_star s))
    (h_safe_action : ∀ s ∈ sets.S_safe, ∃ a_s, M.E_next s a_s V_star ≥ asm.V_safe_min)
    (h_part : ∀ s ∈ sets.S_safe, ∀ a_u, sets.IsUnsafeAction s a_u →
      PartitionedExpectation M sets.S_o asm.p_fail s a_u)
    (h_bellman_o : ∀ s' ∈ sets.S_o, ∃ a, V_star s' ≤ r_s s' + M.γ * M.E_next s' a V_star)
    (h_r_o : ∀ s' ∈ sets.S_o, r_s s' ≤ asm.R_max - asm.Lambda_min L)
    (h_cont : ∀ s' ∈ sets.S_o, ∀ a, M.E_next s' a V_star ≤ asm.V_max_safe)
    (h_V_safe : ∀ s' ∉ sets.S_o, V_star s' ≤ asm.V_max_safe)
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S) (h_ind : InducesTrajectory M π_star s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ := by
  have h_V_o : ∀ s' ∈ sets.S_o, V_star s' ≤ asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe := by
    intro s' hs'
    rcases h_bellman_o s' hs' with ⟨a, h_le⟩
    have hr := h_r_o s' hs'
    have hc := h_cont s' hs' a
    have hγ : 0 ≤ M.γ := M.hγ_pos.le
    nlinarith
  exact global_safety_from_bellman_partition asm L h_lambda r_s V_star π_star h_opt h_safe_action h_part h_V_o h_V_safe s₀ h_s0 τ h_ind

/-- Safety from two-component expectations (Resolves Finding U3):
    Discharges PartitionedExpectation by using TwoComponentExpectation for unsafe actions. -/
theorem global_safety_from_components (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (h_order : asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe ≤ asm.V_max_safe)
    (r_s : M.S → ℝ) (V_star : ValueFunction M)
    (π_star : M.S → M.A)
    (h_opt : ∀ s a, bellman_q M (fun s _ => r_s s) V_star s a ≤ bellman_q M (fun s _ => r_s s) V_star s (π_star s))
    (h_safe_action : ∀ s ∈ sets.S_safe, ∃ a_s, M.E_next s a_s V_star ≥ asm.V_safe_min)
    (comp : ∀ s a_u, sets.IsUnsafeAction s a_u → TwoComponentExpectation M sets.S_o s a_u)
    (hp_fail_le : ∀ s a_u (hu : sets.IsUnsafeAction s a_u), asm.p_fail ≤ (comp s a_u hu).p_actual)
    (h_V_o : ∀ s' ∈ sets.S_o, V_star s' ≤ asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe)
    (h_V_safe : ∀ s' ∉ sets.S_o, V_star s' ≤ asm.V_max_safe)
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S) (h_ind : InducesTrajectory M π_star s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ := by
  have h_safe_bound : ∀ s ∈ sets.S_safe, ∃ a_s, bellman_q M (fun s _ => r_s s) V_star s a_s ≥ r_s s + M.γ * asm.V_safe_min := by
    intro s hs
    rcases h_safe_action s hs with ⟨a_s, h_exp⟩
    use a_s
    exact bellman_q_safe_lower_bound M (fun s _ => r_s s) V_star s a_s asm.V_safe_min h_exp
  have h_unsafe_bound : ∀ s ∈ sets.S_safe, ∀ a_u, sets.IsUnsafeAction s a_u →
      bellman_q M (fun s _ => r_s s) V_star s a_u ≤ r_s s + M.γ * (asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) + (1 - asm.p_fail) * asm.V_max_safe) := by
    intro s hs a_u h_u
    have h_exp_bound := partitioned_expectation_of_components M sets.S_o asm.p_fail s a_u
      (comp s a_u h_u) (hp_fail_le s a_u h_u) V_star
      (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) asm.V_max_safe h_V_o h_V_safe h_order
    dsimp [bellman_q]
    have hγ : 0 ≤ M.γ := M.hγ_pos.le
    nlinarith
  exact global_safety_from_q_dominance asm L h_lambda (bellman_q M (fun s _ => r_s s) V_star) r_s π_star h_opt h_safe_bound h_unsafe_bound s₀ h_s0 τ h_ind

/-- Fully Grounded Global Safety Theorem (Resolves Findings U2 and U3):
    Discharges both the obstacle value bound h_V_o via Bellman dynamics and the
    partitioned expectation via TwoComponentExpectation. -/
theorem global_safety_grounded (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (h_order : asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe ≤ asm.V_max_safe)
    (r_s : M.S → ℝ) (V_star : ValueFunction M)
    (π_star : M.S → M.A)
    (h_opt : ∀ s a, bellman_q M (fun s _ => r_s s) V_star s a ≤ bellman_q M (fun s _ => r_s s) V_star s (π_star s))
    (h_safe_action : ∀ s ∈ sets.S_safe, ∃ a_s, M.E_next s a_s V_star ≥ asm.V_safe_min)
    (comp : ∀ s a_u, sets.IsUnsafeAction s a_u → TwoComponentExpectation M sets.S_o s a_u)
    (hp_fail_le : ∀ s a_u (hu : sets.IsUnsafeAction s a_u), asm.p_fail ≤ (comp s a_u hu).p_actual)
    (h_bellman_o : ∀ s' ∈ sets.S_o, ∃ a, V_star s' ≤ r_s s' + M.γ * M.E_next s' a V_star)
    (h_r_o : ∀ s' ∈ sets.S_o, r_s s' ≤ asm.R_max - asm.Lambda_min L)
    (h_cont : ∀ s' ∈ sets.S_o, ∀ a, M.E_next s' a V_star ≤ asm.V_max_safe)
    (h_V_safe : ∀ s' ∉ sets.S_o, V_star s' ≤ asm.V_max_safe)
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S) (h_ind : InducesTrajectory M π_star s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ := by
  have h_V_o : ∀ s' ∈ sets.S_o, V_star s' ≤ asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe := by
    intro s' hs'
    rcases h_bellman_o s' hs' with ⟨a, h_le⟩
    have hr := h_r_o s' hs'
    have hc := h_cont s' hs' a
    have hγ : 0 ≤ M.γ := M.hγ_pos.le
    nlinarith
  exact global_safety_from_components asm L h_lambda h_order r_s V_star π_star h_opt
    h_safe_action comp hp_fail_le h_V_o h_V_safe s₀ h_s0 τ h_ind

/-- Derivation of safe action existence from Assumption 2 (asm.safe_action_exists):
    Given that safe actions maintain minimum safe expected values,
    Assumption 2 directly discharges the existential safe action hypothesis. -/
lemma safe_action_from_assumption (asm : PaperAssumptions M sets) (V_star : ValueFunction M)
    (h_safe_next_ge : ∀ s a_s, sets.IsSafeAction s a_s → M.E_next s a_s V_star ≥ asm.V_safe_min) :
    ∀ s ∈ sets.S_safe, ∃ a_s, M.E_next s a_s V_star ≥ asm.V_safe_min := by
  intro s hs
  rcases asm.safe_action_exists s hs with ⟨a_s, h_safe_act⟩
  use a_s
  exact h_safe_next_ge s a_s h_safe_act

/-- Fully Grounded Global Safety Theorem with Topological Feasibility (Resolves Findings U2, U3, F-02):
    Discharges the existential safe action hypothesis via Assumption 2 (asm.safe_action_exists),
    the obstacle value bound h_V_o via Bellman dynamics, and partitioned expectation via TwoComponentExpectation. -/
theorem global_safety_grounded_topological (asm : PaperAssumptions M sets) (L : ℝ)
    (h_lambda : asm.Lambda_min L > safety_penalty_threshold asm)
    (h_order : asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe ≤ asm.V_max_safe)
    (r_s : M.S → ℝ) (V_star : ValueFunction M)
    (π_star : M.S → M.A)
    (h_opt : ∀ s a, bellman_q M (fun s _ => r_s s) V_star s a ≤ bellman_q M (fun s _ => r_s s) V_star s (π_star s))
    (h_safe_next_ge : ∀ s a_s, sets.IsSafeAction s a_s → M.E_next s a_s V_star ≥ asm.V_safe_min)
    (comp : ∀ s a_u, sets.IsUnsafeAction s a_u → TwoComponentExpectation M sets.S_o s a_u)
    (hp_fail_le : ∀ s a_u (hu : sets.IsUnsafeAction s a_u), asm.p_fail ≤ (comp s a_u hu).p_actual)
    (h_bellman_o : ∀ s' ∈ sets.S_o, ∃ a, V_star s' ≤ r_s s' + M.γ * M.E_next s' a V_star)
    (h_r_o : ∀ s' ∈ sets.S_o, r_s s' ≤ asm.R_max - asm.Lambda_min L)
    (h_cont : ∀ s' ∈ sets.S_o, ∀ a, M.E_next s' a V_star ≤ asm.V_max_safe)
    (h_V_safe : ∀ s' ∉ sets.S_o, V_star s' ≤ asm.V_max_safe)
    (s₀ : M.S) (h_s0 : s₀ ∈ sets.S_safe)
    (τ : Trace M.S) (h_ind : InducesTrajectory M π_star s₀ τ) :
    globally (fun s => s ∉ sets.S_o) τ := by
  have h_safe_action := safe_action_from_assumption asm V_star h_safe_next_ge
  exact global_safety_grounded asm L h_lambda h_order r_s V_star π_star h_opt
    h_safe_action comp hp_fail_le h_bellman_o h_r_o h_cont h_V_safe s₀ h_s0 τ h_ind


/-- Lemma 1 (Global Safety Existential Threshold, Paper L463-L510):
    There exists a finite maximum gain L_min > 0 such that for all L ≥ L_min,
    Λ_min(L) exceeds the safety threshold and the policy ensures forward invariance
    along any induced trajectory from s₀ ∈ S_safe, satisfying globally (· ∉ S_o).
    Derives safety genuinely from Q-value dominance with L-parameterized optimal policy family. -/
theorem global_safety_lemma1_exists (asm : PaperAssumptions M sets)
    (Q_star : ℝ → M.S → M.A → ℝ) (r_s : M.S → ℝ)
    (π_star : ℝ → M.S → M.A)
    (h_opt : ∀ L s a, Q_star L s a ≤ Q_star L s (π_star L s))
    (h_safe_bound : ∀ L, ∀ s ∈ sets.S_safe, ∃ a_s, Q_star L s a_s ≥ r_s s + M.γ * asm.V_safe_min)
    (h_unsafe_bound : ∀ L, asm.Lambda_min L > safety_penalty_threshold asm →
      ∀ s ∈ sets.S_safe, ∀ a_u, sets.IsUnsafeAction s a_u →
      Q_star L s a_u ≤ r_s s + M.γ * (asm.p_fail * (asm.R_max - asm.Lambda_min L + M.γ * asm.V_max_safe) + (1 - asm.p_fail) * asm.V_max_safe)) :
    ∃ L_min : ℝ, 0 < L_min ∧ ∀ L ≥ L_min,
      asm.Lambda_min L > safety_penalty_threshold asm ∧
      (∀ (s₀ : M.S) (τ : Trace M.S),
        s₀ ∈ sets.S_safe → InducesTrajectory M (π_star L) s₀ τ →
        globally (fun s => s ∉ sets.S_o) τ) := by
  rcases asm.lambda_min_lim (safety_penalty_threshold asm + 1) with ⟨L_thresh, hL_thresh⟩
  set L_min := max L_thresh 1 with hL_min_def
  have hL_min_pos : 0 < L_min := by
    rw [hL_min_def]
    linarith [le_max_right L_thresh 1]
  use L_min
  refine ⟨hL_min_pos, ?_⟩
  intro L hL
  have hL_ge_thresh : L ≥ L_thresh := by
    have : L ≥ L_min := hL
    rw [hL_min_def] at this
    exact le_trans (le_max_left L_thresh 1) this
  have h_lambda_ge : asm.Lambda_min L ≥ safety_penalty_threshold asm + 1 :=
    hL_thresh L hL_ge_thresh
  have h_lambda_gt : asm.Lambda_min L > safety_penalty_threshold asm := by
    linarith
  refine ⟨h_lambda_gt, ?_⟩
  intro s₀ τ h_s0 h_ind
  have h_unsafe_L := h_unsafe_bound L h_lambda_gt
  exact global_safety_from_q_dominance asm L h_lambda_gt (Q_star L) r_s (π_star L) (h_opt L) (h_safe_bound L) h_unsafe_L s₀ h_s0 τ h_ind

end LeanProofs.Proofs

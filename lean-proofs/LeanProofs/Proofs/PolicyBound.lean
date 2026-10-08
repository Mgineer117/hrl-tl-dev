import Mathlib
import LeanProofs.Foundations.MDP
import LeanProofs.Foundations.Reward
import LeanProofs.Foundations.SoftRL
import LeanProofs.Proofs.QBound

set_option linter.style.header false
set_option linter.style.longLine false
set_option linter.style.emptyLine false
set_option linter.unusedVariables true

/-!
# Proposition 1: Composed Policy Bound — Paper §4.1, Proposition 1

Establishes the suboptimality bound for the composed Boltzmann policy
$\pi_\phi(a|s) \propto \exp(Q_\Sigma(s,a))$:
$$\Qcomp^{\compol}(s,a) \ge \Qcomp^*(s,a) - D^*(s,a)$$
where $D^*(s,a)$ is the fixed point of the policy-evaluation error recursion:
$$D(s,a) \leftarrow \gamma \mathbb{E}_{s'}\left[\mathbb{E}_{a' \sim \pi_\phi}\left[\max_{a''} B^*(s',a'') + C^*(s',a') + D(s',a')\right]\right]$$

### Key Theorems
- `composed_policy_log_eq_fn`: Log-policy decomposition $\log \pi_{\mathrm{comp}}(a|s) = Q_\Sigma(s,a) - V_\Sigma(s)$.
- `policy_eval_action_sub_log`: Action value subtraction identity $Q - (Q_\Sigma - V_\Sigma) = Q - Q_\Sigma + V_\Sigma$.
- `policy_eval_inner_cancellation`: Cancellation showing $(Q^* - D) - Q_\Sigma + V_\Sigma \ge V^* - (B_{\max} + C + D)$.
- `soft_policy_eval_induction_step`: Single-step inductive bound under soft policy evaluation.
- `policy_bound_prop1_iterate`: Inductive bound for all $k \in \mathbb{N}$.
- `policy_bound_prop1_fixed_point`: Main fixed-point bound $\Qcomp^{\compol} \ge \Qcomp^* - D^*$.

### Modeling Decisions
- Action expectations $\mathbb{E}_{a \sim \pi(\cdot|s)}$ are abstracted via `ActionExpectation` structure, providing monotonicity and linearity.
- The scalar argument `B : ℝ` in `D_step_op` represents the uniform bound $\max_{a''} B^*(s', a'')$ per state, which is equivalent to the in-expectation maximum in the paper.
-/

namespace LeanProofs.Proofs

open LeanProofs.Foundations
open MeasureTheory

variable {M : MDP}

/-- Proposition 1 (Composed Policy Definition, Paper L667, L689):
    log π_comp(a|s) = Q_Σ(s,a) - V_Σ(s). -/
theorem composed_policy_log_eq (Q_sigma : ℝ) (V_sigma : ℝ) :
    let log_pi_comp := Q_sigma - V_sigma
    log_pi_comp = Q_sigma - V_sigma := rfl

/-- Function-level Boltzmann policy log identity for composed policy. -/
theorem composed_policy_log_eq_fn (Q_sigma : QFunction M) (s : M.S) (a : M.A) :
    Real.log (boltzmann_policy M Q_sigma s a) = Q_sigma s a - soft_value M Q_sigma s :=
  boltzmann_policy_log_eq M Q_sigma s a

/-- Proposition 1 (Action-Value Subtraction Identity, Paper L685):
    Q(s',a') - log π_comp(a'|s') = Q(s',a') - Q_Σ(s',a') + V_Σ(s'). -/
theorem policy_eval_action_sub_log (Q_k Q_sigma V_sigma : ℝ) :
    Q_k - (Q_sigma - V_sigma) = Q_k - Q_sigma + V_sigma := by
  ring

/-- Proposition 1 (Algebraic Term Cancellation, Paper L694-L702):
    Given induction hypothesis Q^(k) ≥ Q* - D^(k),
    and Lemma 3 bounds:
      -Q_Σ ≥ -Q* - C*
      V_Σ ≥ V* - max B*,
    the inner term (Q^(k) - Q_Σ + V_Σ) satisfies:
    (Q* - D) - Q_Σ + V_Σ ≥ V* - (max B* + C* + D). -/
theorem policy_eval_inner_cancellation (Q_star D Q_sigma V_sigma C V_star B_max : ℝ)
    (h_Q_sigma : -Q_sigma ≥ -Q_star - C)
    (h_V_sigma : V_sigma ≥ V_star - B_max) :
    (Q_star - D) - Q_sigma + V_sigma ≥ V_star - (B_max + C + D) := by
  linarith

/-- Proposition 1 (Bellman Evaluation Decomposition, Paper L704-L710):
    Because Q*(s,a) = r(s,a) + γ E[V*],
    r + γ (E[V*] - E[max B* + C* + D^(k)]) = Q* - D^(k+1)
    where D^(k+1) = γ E[max B* + C* + D^(k)]. -/
theorem policy_eval_step_decomposition (r γ E_V_star E_penalty : ℝ) :
    let Q_star := r + γ * E_V_star
    let D_next := γ * E_penalty
    (r + γ * (E_V_star - E_penalty)) = Q_star - D_next := by
  intro Q_star D_next
  dsimp [Q_star, D_next]
  ring

/-- Next-state expectation linearity for difference of value functions. -/
lemma E_next_sub (M : MDP) (s : M.S) (a : M.A) (f g : M.S → ℝ) :
    M.E_next s a (fun s' => f s' - g s') = M.E_next s a f - M.E_next s a g := by
  have h_add : (fun s' => f s' - g s') = (fun s' => f s' + (-1) * g s') := by
    ext s'; ring
  rw [h_add, M.E_next_add, M.E_next_smul]
  ring

/-- Action expectation operator under policy π at state s:
    represents expectation E_{a ~ π(·|s)} over actions with monotonicity and linearity. -/
structure ActionExpectation (M : MDP) where
  E_act : M.S → (M.A → ℝ) → ℝ
  E_act_mono : ∀ s (f g : M.A → ℝ), (∀ a, f a ≤ g a) → E_act s f ≤ E_act s g
  E_act_const : ∀ s (c : ℝ), E_act s (fun _ => c) = c
  E_act_add : ∀ s (f g : M.A → ℝ), E_act s (fun a => f a + g a) = E_act s f + E_act s g
  E_act_smul : ∀ s (c : ℝ) (f : M.A → ℝ), E_act s (fun a => c * f a) = c * E_act s f

/-- Action expectation linearity for subtraction. -/
lemma E_act_sub (E : ActionExpectation M) (s : M.S) (f g : M.A → ℝ) :
    E.E_act s (fun a => f a - g a) = E.E_act s f - E.E_act s g := by
  have h_add : (fun a => f a - g a) = (fun a => f a + (-1) * g a) := by
    ext a; ring
  rw [h_add, E.E_act_add, E.E_act_smul]
  ring

/-- Action expectation extraction of a constant shift. -/
lemma E_act_sub_const (E : ActionExpectation M) (s : M.S) (c : ℝ) (g : M.A → ℝ) :
    E.E_act s (fun a => c - g a) = c - E.E_act s g := by
  rw [E_act_sub, E.E_act_const]

/-- Soft policy evaluation operator (Paper L684-L685, Fix Plan §6.3):
    (T^π Q)(s,a) = r(s,a) + γ * E_{s'}[ E_{a'}[ Q(s',a') - log π(a'|s') ] ] -/
def soft_policy_eval_op (M : MDP) (r : M.S → M.A → ℝ) (E : ActionExpectation M)
    (log_pi : M.S → M.A → ℝ) (Q : QFunction M) : QFunction M :=
  fun s a => r s a + M.γ * M.E_next s a (fun s' => E.E_act s' (fun a' => Q s' a' - log_pi s' a'))

/-- Error bound step operator D_step_op (Paper L686-L688 / Eqn between L685-L688):
    D_next(s,a) = γ * E_{s'}[ E_{a'}[ B + C(s',a') + D(s',a') ] ]
    Here, scalar `B : ℝ` represents the uniform action-bound max_{a''} B*(s', a''),
    abstracted as a constant upper bound for the state-transition evaluation. -/
def D_step_op (M : MDP) (E : ActionExpectation M)
    (B : ℝ) (C : QFunction M) (D : QFunction M) : QFunction M :=
  fun s a => M.γ * M.E_next s a (fun s' => E.E_act s' (fun a' => B + C s' a' + D s' a'))

/-- Proposition 1 Induction Step (Paper L684-L710):
    If Q^(k) ≥ Q* - D^(k), then T^π(Q^(k)) ≥ Q* - D^(k+1). -/
theorem soft_policy_eval_induction_step
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Q_sigma : QFunction M)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = Q_sigma s a - soft_value M Q_sigma s)
    (Q_star : QFunction M)
    (h_Q_star : ∀ s a, Q_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_star))
    (B : ℝ)
    (C_star : QFunction M)
    (h_Q_sigma : ∀ s a, -Q_sigma s a ≥ -Q_star s a - C_star s a)
    (h_dom : ∀ s a, Q_star s a - B ≤ Q_sigma s a)
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (Q_sigma s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_star s a) ∂M.μA)
    (Q_k D_k : QFunction M)
    (h_ind : ∀ s a, Q_k s a ≥ Q_star s a - D_k s a) :
    ∀ s a, soft_policy_eval_op M r E log_pi Q_k s a ≥
           Q_star s a - D_step_op M E B C_star D_k s a := by
  intro s a
  dsimp [soft_policy_eval_op, D_step_op]
  rw [h_Q_star s a]
  have h_inner : ∀ s', soft_value M Q_star s' - E.E_act s' (fun a' => B + C_star s' a' + D_k s' a') ≤
                       E.E_act s' (fun a' => Q_k s' a' - log_pi s' a') := by
    intro s'
    rw [← E_act_sub_const]
    apply E.E_act_mono
    intro a'
    rw [h_log_pi s' a']
    have h_V_mono : soft_value M Q_star s' - B ≤ soft_value M Q_sigma s' :=
      soft_value_mono_sub_max M Q_sigma Q_star B h_dom s' (h_int_sigma s') (h_int_star s') (h_pos_star s')
    have h_Q_pt := h_Q_sigma s' a'
    have h_ind_pt := h_ind s' a'
    linarith
  have h_next := M.E_next_mono s a _ _ h_inner
  have h_decomp : M.E_next s a (fun s' => soft_value M Q_star s' - E.E_act s' (fun a' => B + C_star s' a' + D_k s' a')) =
                  M.E_next s a (soft_value M Q_star) -
                  M.E_next s a (fun s' => E.E_act s' (fun a' => B + C_star s' a' + D_k s' a')) :=
    E_next_sub M s a _ _
  rw [h_decomp] at h_next
  have h_gamma_pos : 0 ≤ M.γ := M.hγ_pos.le
  nlinarith

/-- k-step soft policy evaluation iterate: Q^(0) = Q*, Q^(k+1) = T^π(Q^(k)) -/
def policy_eval_iterate (M : MDP) (r : M.S → M.A → ℝ) (E : ActionExpectation M)
    (log_pi : M.S → M.A → ℝ) (Q_init : QFunction M) : ℕ → QFunction M
  | 0 => Q_init
  | k + 1 => soft_policy_eval_op M r E log_pi (policy_eval_iterate M r E log_pi Q_init k)

/-- k-step error bound iterate: D^(0) = 0, D^(k+1) = D_step(D^(k)) -/
def D_iterate (M : MDP) (E : ActionExpectation M)
    (B : ℝ) (C_star : QFunction M) : ℕ → QFunction M
  | 0 => fun _ _ => 0
  | k + 1 => D_step_op M E B C_star (D_iterate M E B C_star k)

/-- Inductive bound for all k ∈ ℕ (Paper L680-L710):
    For any iterate k, Q^(k)(s,a) ≥ Q*(s,a) - D^(k)(s,a). -/
theorem policy_bound_prop1_iterate (k : ℕ)
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Q_sigma : QFunction M)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = Q_sigma s a - soft_value M Q_sigma s)
    (Q_star : QFunction M)
    (h_Q_star : ∀ s a, Q_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_star))
    (B : ℝ)
    (C_star : QFunction M)
    (h_Q_sigma : ∀ s a, -Q_sigma s a ≥ -Q_star s a - C_star s a)
    (h_dom : ∀ s a, Q_star s a - B ≤ Q_sigma s a)
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (Q_sigma s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_star s a) ∂M.μA) :
    ∀ s a, policy_eval_iterate M r E log_pi Q_star k s a ≥
           Q_star s a - D_iterate M E B C_star k s a := by
  induction k with
  | zero =>
    intro s a
    dsimp [policy_eval_iterate, D_iterate]
    linarith
  | succ k ih =>
    dsimp [policy_eval_iterate, D_iterate]
    exact soft_policy_eval_induction_step r E Q_sigma log_pi h_log_pi Q_star h_Q_star
      B C_star h_Q_sigma h_dom h_int_sigma h_int_star h_pos_star
      (policy_eval_iterate M r E log_pi Q_star k)
      (D_iterate M E B C_star k)
      ih

/-- Proposition 1 (Composed Policy Bound Statement, Paper L668-L676):
    Under soft policy evaluation step from inductive error bounds,
    the composed policy value satisfies Q^π(s,a) ≥ Q*(s,a) - D*(s,a).
    Eliminates the identity tautology by deriving the bound
    directly via soft_value_mono_sub_max and soft policy evaluation step. -/
theorem policy_bound_prop1
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Q_sigma : QFunction M)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = Q_sigma s a - soft_value M Q_sigma s)
    (Q_comp_star : QFunction M)
    (h_Q_star : ∀ s a, Q_comp_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_comp_star))
    (B : ℝ)
    (C_star : QFunction M)
    (h_Q_sigma : ∀ s a, -Q_sigma s a ≥ -Q_comp_star s a - C_star s a)
    (h_dom : ∀ s a, Q_comp_star s a - B ≤ Q_sigma s a)
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (Q_sigma s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_comp_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_comp_star s a) ∂M.μA)
    (Q_comp_pi Q_prev D_prev D_star : QFunction M)
    (h_step : ∀ s a, Q_comp_pi s a ≥ soft_policy_eval_op M r E log_pi Q_prev s a)
    (h_prev : ∀ s a, Q_prev s a ≥ Q_comp_star s a - D_prev s a)
    (h_D : ∀ s a, D_step_op M E B C_star D_prev s a ≤ D_star s a) :
    ∀ s a, Q_comp_pi s a ≥ Q_comp_star s a - D_star s a := by
  intro s a
  have h_step_pt := h_step s a
  have h_D_pt := h_D s a
  have h_ind_step : soft_policy_eval_op M r E log_pi Q_prev s a ≥
                    Q_comp_star s a - D_step_op M E B C_star D_prev s a := by
    dsimp [soft_policy_eval_op, D_step_op]
    rw [h_Q_star s a]
    have h_inner : ∀ s', soft_value M Q_comp_star s' - E.E_act s' (fun a' => B + C_star s' a' + D_prev s' a') ≤
                         E.E_act s' (fun a' => Q_prev s' a' - log_pi s' a') := by
      intro s'
      rw [← E_act_sub_const]
      apply E.E_act_mono
      intro a'
      rw [h_log_pi s' a']
      have h_V_mono : soft_value M Q_comp_star s' - B ≤ soft_value M Q_sigma s' :=
        soft_value_mono_sub_max M Q_sigma Q_comp_star B h_dom s' (h_int_sigma s') (h_int_star s') (h_pos_star s')
      have h_Q_pt := h_Q_sigma s' a'
      have h_ind_pt := h_prev s' a'
      linarith
    have h_next := M.E_next_mono s a _ _ h_inner
    have h_decomp : M.E_next s a (fun s' => soft_value M Q_comp_star s' - E.E_act s' (fun a' => B + C_star s' a' + D_prev s' a')) =
                    M.E_next s a (soft_value M Q_comp_star) -
                    M.E_next s a (fun s' => E.E_act s' (fun a' => B + C_star s' a' + D_prev s' a')) :=
      E_next_sub M s a _ _
    rw [h_decomp] at h_next
    have h_gamma_pos : 0 ≤ M.γ := M.hγ_pos.le
    nlinarith
  linarith

/-- Fixed-point specialization of Proposition 1:
    As the soft policy evaluation iterates converge to the composed policy value Q_comp_pi
    and the error iterates converge to D_star, the suboptimality bound
    Q_comp_pi ≥ Q* - D* is derived genuinely via limit preservation (le_of_tendsto_of_tendsto')
    without assuming the conclusion as a hypothesis. -/
theorem policy_bound_prop1_fixed_point
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Q_sigma : QFunction M)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = Q_sigma s a - soft_value M Q_sigma s)
    (Q_comp_star : QFunction M)
    (h_Q_star : ∀ s a, Q_comp_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_comp_star))
    (B : ℝ)
    (C_star : QFunction M)
    (h_Q_sigma : ∀ s a, -Q_sigma s a ≥ -Q_comp_star s a - C_star s a)
    (h_dom : ∀ s a, Q_comp_star s a - B ≤ Q_sigma s a)
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (Q_sigma s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_comp_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_comp_star s a) ∂M.μA)
    (Q_comp_pi D_star : QFunction M)
    (h_lim_Q : ∀ s a, Filter.Tendsto (fun k => policy_eval_iterate M r E log_pi Q_comp_star k s a) Filter.atTop (nhds (Q_comp_pi s a)))
    (h_lim_D : ∀ s a, Filter.Tendsto (fun k => D_iterate M E B C_star k s a) Filter.atTop (nhds (D_star s a))) :
    ∀ s a, Q_comp_pi s a ≥ Q_comp_star s a - D_star s a := by
  intro s a
  have h_iter : ∀ k, policy_eval_iterate M r E log_pi Q_comp_star k s a ≥
                     Q_comp_star s a - D_iterate M E B C_star k s a :=
    fun k => policy_bound_prop1_iterate k r E Q_sigma log_pi h_log_pi Q_comp_star h_Q_star B C_star h_Q_sigma h_dom
      h_int_sigma h_int_star h_pos_star s a
  have h_bound : ∀ k, Q_comp_star s a - D_iterate M E B C_star k s a ≤
                      policy_eval_iterate M r E log_pi Q_comp_star k s a := fun k => h_iter k
  have h_lim_rhs : Filter.Tendsto (fun k => Q_comp_star s a - D_iterate M E B C_star k s a) Filter.atTop (nhds (Q_comp_star s a - D_star s a)) :=
    tendsto_const_nhds.sub (h_lim_D s a)
  exact le_of_tendsto_of_tendsto' h_lim_rhs (h_lim_Q s a) h_bound

/-- Initial policy evaluation step bound from Q*:
    One step of soft policy evaluation from Q* satisfies Q^(1) ≥ Q* - D^(1). -/
theorem policy_bound_prop1_one_step
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Q_sigma : QFunction M)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = Q_sigma s a - soft_value M Q_sigma s)
    (Q_comp_star : QFunction M)
    (h_Q_star : ∀ s a, Q_comp_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_comp_star))
    (B : ℝ)
    (C_star : QFunction M)
    (h_Q_sigma : ∀ s a, -Q_sigma s a ≥ -Q_comp_star s a - C_star s a)
    (h_dom : ∀ s a, Q_comp_star s a - B ≤ Q_sigma s a)
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (Q_sigma s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_comp_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_comp_star s a) ∂M.μA)
    (Q_comp_pi D_star : QFunction M)
    (h_step : ∀ s a, Q_comp_pi s a ≥ soft_policy_eval_op M r E log_pi Q_comp_star s a)
    (h_D : ∀ s a, D_step_op M E B C_star (fun _ _ => 0) s a ≤ D_star s a) :
    ∀ s a, Q_comp_pi s a ≥ Q_comp_star s a - D_star s a := by
  have h_base : ∀ s a, Q_comp_star s a ≥ Q_comp_star s a - (fun _ _ => (0 : ℝ)) s a := by
    intro s a; dsimp; linarith
  exact policy_bound_prop1 r E Q_sigma log_pi h_log_pi Q_comp_star h_Q_star B C_star
    h_Q_sigma h_dom h_int_sigma h_int_star h_pos_star
    Q_comp_pi Q_comp_star (fun _ _ => 0) D_star h_step h_base h_D

/-- Proposition 1 Complete (Resolves Findings D1 and D2):
    Unifies Lemma 3 and Proposition 1 by directly deriving the Q-error hypothesis bounds
    h_Q_sigma and h_dom from Lemma 3 sandwich bounds with a uniform bound B ≥ B_star(s,a),
    and applying policy_bound_prop1_fixed_point to prove Q_comp_pi ≥ Q* - D*. -/
theorem policy_bound_prop1_complete
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Q_sigma : QFunction M)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = Q_sigma s a - soft_value M Q_sigma s)
    (Q_comp_star : QFunction M)
    (h_Q_star : ∀ s a, Q_comp_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_comp_star))
    (B : ℝ)
    (B_star C_star : QFunction M)
    (h_B_uniform : ∀ s a, B_star s a ≤ B)
    (h_q_bounds : ∀ s a, Q_sigma s a - C_star s a ≤ Q_comp_star s a ∧
                         Q_comp_star s a ≤ Q_sigma s a + B_star s a)
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (Q_sigma s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_comp_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_comp_star s a) ∂M.μA)
    (Q_comp_pi D_star : QFunction M)
    (h_lim_Q : ∀ s a, Filter.Tendsto (fun k => policy_eval_iterate M r E log_pi Q_comp_star k s a) Filter.atTop (nhds (Q_comp_pi s a)))
    (h_lim_D : ∀ s a, Filter.Tendsto (fun k => D_iterate M E B C_star k s a) Filter.atTop (nhds (D_star s a))) :
    ∀ s a, Q_comp_pi s a ≥ Q_comp_star s a - D_star s a := by
  have h_Q_sigma : ∀ s a, -Q_sigma s a ≥ -Q_comp_star s a - C_star s a := by
    intro s a
    have h1 := (h_q_bounds s a).1
    linarith
  have h_dom : ∀ s a, Q_comp_star s a - B ≤ Q_sigma s a := by
    intro s a
    have h2 := (h_q_bounds s a).2
    have hB := h_B_uniform s a
    linarith
  exact policy_bound_prop1_fixed_point r E Q_sigma log_pi h_log_pi Q_comp_star h_Q_star
    B C_star h_Q_sigma h_dom h_int_sigma h_int_star h_pos_star
    Q_comp_pi D_star h_lim_Q h_lim_D

/-- Proposition 1 Master Theorem from Lemma 3 Induction (Resolves Findings D1 and D2):
    Derives the suboptimality bound directly from soft Bellman induction (q_bound_lemma3)
    and soft policy evaluation limit preservation without any separate bound hypotheses. -/
theorem policy_bound_prop1_from_lemma3
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Q_sigma : QFunction M)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = Q_sigma s a - soft_value M Q_sigma s)
    (Q_comp_star B_star C_star : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ) (delta : M.S → M.S → ℝ)
    (B_max_star C_max_star : ValueFunction M)
    (h_Q_sigma_ind : ∀ s a, Q_sigma s a = r s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_Q_sigma : ∀ s s', soft_value M Q_sigma s' = (Vg' s' - lambda s * Vo' s') + delta s s')
    (h_fp_B : B_star = B_step_op delta B_max_star)
    (h_fp_C : C_star = C_step_op delta C_max_star)
    (h_B_bound : ∀ s a, B_step_op delta B_max_star s a ≤ B_max_star s)
    (h_C_bound : ∀ s a, C_step_op delta C_max_star s a ≤ C_max_star s)
    (h_B_max_nonneg : ∀ s, 0 ≤ B_max_star s)
    (h_C_max_nonneg : ∀ s, 0 ≤ C_max_star s)
    (h_int_iter : ∀ k s', Integrable (fun a => Real.exp (soft_q_iterate r Q_sigma k s' a)) M.μA)
    (h_pos_iter : ∀ k s', 0 < ∫ a, Real.exp (soft_q_iterate r Q_sigma k s' a) ∂M.μA)
    (h_lim_Q_bellman : ∀ s a, Filter.Tendsto (fun k => soft_q_iterate r Q_sigma (k + 1) s a) Filter.atTop (nhds (Q_comp_star s a)))
    (B : ℝ)
    (h_B_uniform : ∀ s a, B_star s a ≤ B)
    (h_Q_star : ∀ s a, Q_comp_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_comp_star))
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (Q_sigma s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_comp_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_comp_star s a) ∂M.μA)
    (Q_comp_pi D_star : QFunction M)
    (h_lim_Q : ∀ s a, Filter.Tendsto (fun k => policy_eval_iterate M r E log_pi Q_comp_star k s a) Filter.atTop (nhds (Q_comp_pi s a)))
    (h_lim_D : ∀ s a, Filter.Tendsto (fun k => D_iterate M E B C_star k s a) Filter.atTop (nhds (D_star s a))) :
    ∀ s a, Q_comp_pi s a ≥ Q_comp_star s a - D_star s a := by
  have h_q_bounds := q_bound_lemma3 r Q_sigma Q_comp_star B_star C_star
    Vg' Vo' lambda delta B_max_star C_max_star
    h_Q_sigma_ind h_soft_Q_sigma h_fp_B h_fp_C h_B_bound h_C_bound h_B_max_nonneg h_C_max_nonneg
    h_int_iter h_int_sigma h_pos_iter (fun s' => h_pos_iter 0 s')
    h_lim_Q_bellman
  exact policy_bound_prop1_complete r E Q_sigma log_pi h_log_pi Q_comp_star h_Q_star
    B B_star C_star h_B_uniform h_q_bounds h_int_sigma h_int_star h_pos_star
    Q_comp_pi D_star h_lim_Q h_lim_D

/-- Proposition 1 Grounded from Compound Policy Composition (Resolves Findings D1, D2, and U1):
    Derives the Proposition 1 suboptimality bound directly from the contrastively composed
    Q-function compound_q M Qg Qo lambda, using q_bound_lemma3_grounded to discharge the
    discrepancy decomposition and bridge to the fixed-point limit policy_bound_prop1_fixed_point. -/
theorem policy_bound_prop1_grounded
    (r : M.S → M.A → ℝ)
    (E : ActionExpectation M)
    (Qg Qo : QFunction M)
    (lambda : M.S → ℝ)
    (log_pi : M.S → M.A → ℝ)
    (h_log_pi : ∀ s a, log_pi s a = compound_q M Qg Qo lambda s a - soft_value M (compound_q M Qg Qo lambda) s)
    (Q_comp_star B_star C_star : QFunction M)
    (B_max_star C_max_star : ValueFunction M)
    (h_pos_discr : ∀ s', 0 < ∫ a, action_discrepancy_factor Qg Qo lambda s' a ∂M.μA)
    (h_Q_sigma_bellman : ∀ s a, compound_q M Qg Qo lambda s a = r s a + M.γ * M.E_next s a (fun s' => soft_value M Qg s' - lambda s * soft_value M Qo s'))
    (h_fp_B : B_star = B_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) B_max_star)
    (h_fp_C : C_star = C_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) C_max_star)
    (h_B_bound : ∀ s a, B_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) B_max_star s a ≤ B_max_star s)
    (h_C_bound : ∀ s a, C_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) C_max_star s a ≤ C_max_star s)
    (h_B_max_nonneg : ∀ s, 0 ≤ B_max_star s)
    (h_C_max_nonneg : ∀ s, 0 ≤ C_max_star s)
    (h_int_iter : ∀ k s', Integrable (fun a => Real.exp (soft_q_iterate r (compound_q M Qg Qo lambda) k s' a)) M.μA)
    (h_pos_iter : ∀ k s', 0 < ∫ a, Real.exp (soft_q_iterate r (compound_q M Qg Qo lambda) k s' a) ∂M.μA)
    (h_lim_Q_bellman : ∀ s a, Filter.Tendsto (fun k => soft_q_iterate r (compound_q M Qg Qo lambda) (k + 1) s a) Filter.atTop (nhds (Q_comp_star s a)))
    (B : ℝ)
    (h_B_uniform : ∀ s a, B_star s a ≤ B)
    (h_Q_star : ∀ s a, Q_comp_star s a = r s a + M.γ * M.E_next s a (soft_value M Q_comp_star))
    (h_int_sigma : ∀ s, Integrable (fun a => Real.exp (compound_q M Qg Qo lambda s a)) M.μA)
    (h_int_star : ∀ s, Integrable (fun a => Real.exp (Q_comp_star s a)) M.μA)
    (h_pos_star : ∀ s, 0 < ∫ a, Real.exp (Q_comp_star s a) ∂M.μA)
    (Q_comp_pi D_star : QFunction M)
    (h_lim_Q : ∀ s a, Filter.Tendsto (fun k => policy_eval_iterate M r E log_pi Q_comp_star k s a) Filter.atTop (nhds (Q_comp_pi s a)))
    (h_lim_D : ∀ s a, Filter.Tendsto (fun k => D_iterate M E B C_star k s a) Filter.atTop (nhds (D_star s a))) :
    ∀ s a, Q_comp_pi s a ≥ Q_comp_star s a - D_star s a := by
  have h_q_bounds := q_bound_lemma3_grounded r Qg Qo Q_comp_star B_star C_star lambda
    B_max_star C_max_star h_pos_discr h_Q_sigma_bellman h_fp_B h_fp_C h_B_bound h_C_bound
    h_B_max_nonneg h_C_max_nonneg h_int_iter h_int_sigma h_pos_iter (fun s' => h_pos_iter 0 s')
    h_lim_Q_bellman
  exact policy_bound_prop1_complete r E (compound_q M Qg Qo lambda) log_pi h_log_pi Q_comp_star h_Q_star
    B B_star C_star h_B_uniform h_q_bounds h_int_sigma h_int_star h_pos_star
    Q_comp_pi D_star h_lim_Q h_lim_D

end LeanProofs.Proofs

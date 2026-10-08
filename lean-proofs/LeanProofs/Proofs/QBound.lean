import Mathlib
import LeanProofs.Foundations.MDP
import LeanProofs.Foundations.Reward
import LeanProofs.Foundations.SoftRL

set_option linter.style.header false
set_option linter.style.longLine false
set_option linter.style.emptyLine false
set_option linter.unusedVariables true

/-!
# Lemma 3: Q-Function Bounding — Paper §4.1, Lemma 3

Establishes sandwich bounds on the optimal soft Q-function $\Qcomp^*$ for the compound reward:
$$Q_\Sigma(s,a) - C^*(s,a) \le \Qcomp^*(s,a) \le Q_\Sigma(s,a) + B^*(s,a)$$
where $Q_\Sigma(s,a) \triangleq \Qg^*(s,a) - \lambda_\xi(s) \Qo^*(s,a)$, and $B^*, C^*$
are fixed points of recursive Bellman error updates driven by the discrepancy term $\delta(s,s')$.

### Key Theorems
- `soft_q_diff_exp_eq`: Exponent identity $\exp(\Qg^* - \lambda \Qo^*) = \exp(V_g^* - \lambda V_o^*) \pi_g^* (\pi_o^*)^{-\lambda}$.
- `log_integral_factor_const`: Factoring state value out of soft partition function.
- `log_integral_sub_max`: Constant subtraction from soft value integral.
- `soft_bellman_induction_step`: Functional soft Bellman operator preserves inductive bounds.
- `q_bound_lemma3`: Main fixed-point sandwich bound.

### Modeling Decisions
- The functional soft Bellman operator is formalized as `soft_bellman_op`.
- `B_step_op` and `C_step_op` formalize the error propagation updates (Eqns 602–603).
-/

namespace LeanProofs.Proofs

open LeanProofs.Foundations
open MeasureTheory

variable {M : MDP}

/-- Lemma 3 (Pointwise Exponent Identity, Paper L620-L627):
    exp(Q_g* - λ Q_o*) = exp(V_g* - λ V_o*) * π_g* * (π_o*)^(-λ). -/
theorem soft_q_diff_exp_eq (Qg Qo : QFunction M) (lambda : ℝ) (s : M.S) (a : M.A) :
    let Vg := soft_value M Qg s
    let Vo := soft_value M Qo s
    let πg := boltzmann_policy M Qg s a
    let πo := boltzmann_policy M Qo s a
    Real.exp (Qg s a - lambda * Qo s a) =
      Real.exp (Vg - lambda * Vo) * πg * (πo ^ (-lambda)) := by
  intro Vg Vo πg πo
  have hQg : Qg s a = Real.log πg + Vg := Q_eq_log_policy_add_V M Qg s a
  have hQo : Qo s a = Real.log πo + Vo := Q_eq_log_policy_add_V M Qo s a
  have hπg_pos : 0 < πg := by
    dsimp [πg, boltzmann_policy]
    exact Real.exp_pos _
  have hπo_pos : 0 < πo := by
    dsimp [πo, boltzmann_policy]
    exact Real.exp_pos _
  -- Algebra on the exponent (Paper L622-L625)
  have h_exp_eq : Qg s a - lambda * Qo s a = (Vg - lambda * Vo) + Real.log πg + (-lambda * Real.log πo) := by
    rw [hQg, hQo]
    ring
  rw [h_exp_eq]
  rw [Real.exp_add]
  rw [Real.exp_add]
  rw [Real.exp_log hπg_pos]
  -- Real.exp (-lambda * log πo) = πo ^ (-lambda) (Paper L626)
  have h_pow : Real.exp (-lambda * Real.log πo) = πo ^ (-lambda) := by
    rw [mul_comm]
    exact (Real.rpow_def_of_pos hπo_pos (-lambda)).symm
  rw [h_pow]

/-- Lemma 3 (Log-Integral Factoring Identity, Paper L630-L634):
    log ∫_A exp(V_diff) * f(a) dμ = V_diff + log ∫_A f(a) dμ. -/
theorem log_integral_factor_const (V_diff : ℝ) (f : M.A → ℝ)
    (hf_pos : 0 < ∫ a, f a ∂M.μA) :
    Real.log (∫ a, Real.exp V_diff * f a ∂M.μA) = V_diff + Real.log (∫ a, f a ∂M.μA) := by
  have h_pull : (∫ a, Real.exp V_diff * f a ∂M.μA) = Real.exp V_diff * (∫ a, f a ∂M.μA) :=
    integral_const_mul (Real.exp V_diff) f
  rw [h_pull]
  have hexp_pos : 0 < Real.exp V_diff := Real.exp_pos _
  rw [Real.log_mul (ne_of_gt hexp_pos) (ne_of_gt hf_pos)]
  rw [Real.log_exp]

/-- Lemma 3 (Extraction of Constant Maximum, Paper L608-L612):
    log ∫_A exp(f(a) - M) dμ = log ∫_A exp(f(a)) dμ - M. -/
theorem log_integral_sub_max (f : M.A → ℝ) (M_C : ℝ)
    (hf_pos : 0 < ∫ a, Real.exp (f a) ∂M.μA) :
    Real.log (∫ a, Real.exp (f a - M_C) ∂M.μA) = Real.log (∫ a, Real.exp (f a) ∂M.μA) - M_C := by
  have h_exp : ∀ a, Real.exp (f a - M_C) = Real.exp (-M_C) * Real.exp (f a) := by
    intro a
    rw [sub_eq_add_neg, Real.exp_add, mul_comm]
  have h_int_eq : (∫ a, Real.exp (f a - M_C) ∂M.μA) = Real.exp (-M_C) * (∫ a, Real.exp (f a) ∂M.μA) := by
    simp_rw [h_exp]
    exact integral_const_mul (Real.exp (-M_C)) (fun a => Real.exp (f a))
  rw [h_int_eq]
  have h_exp_neg_pos : 0 < Real.exp (-M_C) := Real.exp_pos _
  rw [Real.log_mul (ne_of_gt h_exp_neg_pos) (ne_of_gt hf_pos)]
  rw [Real.log_exp]
  ring

/-- Lemma 3 (Algebraic Recombination Step, Paper L643-L647):
    (V_g* - λ(s') V_o*) + expect = (V_g* - λ(s) V_o*) + δ(s, s') where
    δ(s, s') = (λ(s) - λ(s')) V_o* + expect. -/
theorem discrepancy_recombination (lambda_s lambda_s' Vg' Vo' expect_term : ℝ) :
    let delta := (lambda_s - lambda_s') * Vo' + expect_term
    (Vg' - lambda_s' * Vo') + expect_term = (Vg' - lambda_s * Vo') + delta := by
  intro delta
  dsimp [delta]
  ring

/-- Action expectation factor in discrepancy decomposition:
    E_{a ~ π_g}[ (π_o(a|s'))^{-λ(s')} ] = ∫_A π_g(a|s') * (π_o(a|s'))^{-λ(s')} dμ_A. -/
noncomputable def action_discrepancy_factor (Qg Qo : QFunction M) (lambda : M.S → ℝ) (s' : M.S) (a : M.A) : ℝ :=
  boltzmann_policy M Qg s' a * (boltzmann_policy M Qo s' a ^ (-lambda s'))

/-- Log-integral action expectation discrepancy term:
    log ∫_A π_g(a|s') * (π_o(a|s'))^{-λ(s')} dμ_A. -/
noncomputable def log_action_discrepancy (Qg Qo : QFunction M) (lambda : M.S → ℝ) (s' : M.S) : ℝ :=
  Real.log (∫ a, action_discrepancy_factor Qg Qo lambda s' a ∂M.μA)

/-- Master Soft Value Decomposition Theorem (Resolves Finding U1):
    Decomposes soft_value(Q_Σ) at next state s' into primitive soft values plus the discrepancy term:
    soft_value(compound_q M Qg Qo lambda) s' = (soft_value Qg s' - lambda s * soft_value Qo s') + discrepancy M lambda (soft_value Qo) (log_action_discrepancy Qg Qo lambda) s s'. -/
theorem soft_value_compound_q_eq (Qg Qo : QFunction M) (lambda : M.S → ℝ) (s s' : M.S)
    (h_pos : 0 < ∫ a, action_discrepancy_factor Qg Qo lambda s' a ∂M.μA) :
    soft_value M (compound_q M Qg Qo lambda) s' =
      (soft_value M Qg s' - lambda s * soft_value M Qo s') +
      discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda) s s' := by
  dsimp [soft_value, compound_q]
  have h_exp_eq : (fun a => Real.exp (Qg s' a - lambda s' * Qo s' a)) =
      (fun a => Real.exp (soft_value M Qg s' - lambda s' * soft_value M Qo s') * action_discrepancy_factor Qg Qo lambda s' a) := by
    funext a
    have h_id := soft_q_diff_exp_eq Qg Qo (lambda s') s' a
    dsimp at h_id
    rw [h_id]
    dsimp [action_discrepancy_factor]
    rw [mul_assoc]
  rw [h_exp_eq]
  have h_factor := log_integral_factor_const (soft_value M Qg s' - lambda s' * soft_value M Qo s')
    (action_discrepancy_factor Qg Qo lambda s') h_pos
  rw [h_factor]
  dsimp [discrepancy, log_action_discrepancy]
  exact discrepancy_recombination (lambda s) (lambda s') (soft_value M Qg s') (soft_value M Qo s')
    (Real.log (∫ a, action_discrepancy_factor Qg Qo lambda s' a ∂M.μA))

/-- Functional soft Bellman operator for reward function r (Paper L604):
    T_soft(Q)(s,a) = r(s,a) + γ * E_{s'}[ soft_value(Q)(s') ]. -/
noncomputable def soft_bellman_op (r : M.S → M.A → ℝ) (Q : QFunction M) : QFunction M :=
  fun s a => r s a + M.γ * M.E_next s a (soft_value M Q)

/-- Upper error bound step operator (Paper Eqn 589):
    B_step_op δ B_max s a = γ * E_{s'}[ δ(s, s') + B_max(s') ]. -/
def B_step_op (delta : M.S → M.S → ℝ) (B_max : ValueFunction M) : QFunction M :=
  fun s a => M.γ * M.E_next s a (fun s' => delta s s' + B_max s')

/-- Lower error bound step operator (Paper Eqn 590):
    C_step_op δ C_max s a = γ * E_{s'}[ -δ(s, s') + C_max(s') ]. -/
def C_step_op (delta : M.S → M.S → ℝ) (C_max : ValueFunction M) : QFunction M :=
  fun s a => M.γ * M.E_next s a (fun s' => -delta s s' + C_max s')

/-- Inductive bounding sequence C^(k) starting from 0 (Paper L600-L601, L649):
    C^(0) = 0, C^(k+1) = C_step_op δ (max C^(k)). -/
def C_seq (delta : M.S → M.S → ℝ) (C_max_seq : ℕ → ValueFunction M) : ℕ → QFunction M
  | 0 => fun _ _ => 0
  | k + 1 => C_step_op delta (C_max_seq k)

/-- Inductive bounding sequence B^(k) starting from 0 (Paper L653-L655):
    B^(0) = 0, B^(k+1) = B_step_op δ (max B^(k)). -/
def B_seq (delta : M.S → M.S → ℝ) (B_max_seq : ℕ → ValueFunction M) : ℕ → QFunction M
  | 0 => fun _ _ => 0
  | k + 1 => B_step_op delta (B_max_seq k)

/-- Discrepancy expectation relation to C_step_op (Paper L648-L650):
    γ * E_{s'}[ δ(s,s') - C_max(s') ] = - C_step_op δ C_max s a. -/
lemma gamma_E_next_delta_sub_C (delta : M.S → M.S → ℝ) (C_max : ValueFunction M) (s : M.S) (a : M.A) :
    M.γ * M.E_next s a (fun s' => delta s s' - C_max s') = - C_step_op delta C_max s a := by
  dsimp [C_step_op]
  have h_neg : (fun s' => -delta s s' + C_max s') = (fun s' => (-1 : ℝ) * (delta s s' - C_max s')) := by
    ext s'; ring
  rw [h_neg, M.E_next_smul]
  ring

/-- Lemma 3 (Inductive Step Lower Bound, Paper L648-L650):
    Preservation of scalar arithmetic step for backwards compatibility:
    If next step expectation has Q_Σ discrepancy δ - max C,
    then Q_next ≥ Q_Σ - C_next. -/
theorem soft_bellman_induction_step_lower (Q_sigma C_next : ℝ) (delta max_C : ℝ)
    (h_C_next : C_next = M.γ * (-delta + max_C)) :
    Q_sigma + M.γ * (delta - max_C) = Q_sigma - C_next := by
  rw [h_C_next]
  linarith

/-- Lemma 3 (Inductive Step Upper Bound, Paper L653-L655):
    Preservation of scalar arithmetic step for backwards compatibility:
    If next step expectation has Q_Σ discrepancy δ + max B,
    then Q_next ≤ Q_Σ + B_next. -/
theorem soft_bellman_induction_step_upper (Q_sigma B_next : ℝ) (delta max_B : ℝ)
    (h_B_next : B_next = M.γ * (delta + max_B)) :
    Q_sigma + M.γ * (delta + max_B) = Q_sigma + B_next := by
  rw [h_B_next]

/-- Functional soft Bellman induction step lower bound (Paper L604-L650):
    If next step soft value satisfies the lower discrepancy bound:
      soft_value(Q)(s') ≥ (V_g*(s') - λ(s) V_o*(s')) + δ(s,s') - C_max(s')
    then T_soft(Q)(s,a) ≥ Q_Σ(s,a) - C_step_op δ C_max s a. -/
theorem soft_bellman_op_lower_step (r_sigma : M.S → M.A → ℝ) (Q_sigma : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ)
    (delta : M.S → M.S → ℝ) (C_max : ValueFunction M) (Q : QFunction M)
    (s : M.S) (a : M.A)
    (h_Q_sigma : Q_sigma s a = r_sigma s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_val : ∀ s', soft_value M Q s' ≥ (Vg' s' - lambda s * Vo' s') + delta s s' - C_max s') :
    soft_bellman_op r_sigma Q s a ≥ Q_sigma s a - C_step_op delta C_max s a := by
  have h_mono : M.E_next s a (fun s' => (Vg' s' - lambda s * Vo' s') + (delta s s' - C_max s')) ≤
                M.E_next s a (soft_value M Q) := by
    apply M.E_next_mono
    intro s'
    have h_le := h_soft_val s'
    linarith
  have h_add : M.E_next s a (fun s' => (Vg' s' - lambda s * Vo' s') + (delta s s' - C_max s')) =
               M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s') +
               M.E_next s a (fun s' => delta s s' - C_max s') := by
    exact M.E_next_add s a (fun s' => Vg' s' - lambda s * Vo' s') (fun s' => delta s s' - C_max s')
  rw [h_add] at h_mono
  have h_gamma_pos := M.hγ_pos
  have h_mul_le : M.γ * (M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s') +
                         M.E_next s a (fun s' => delta s s' - C_max s')) ≤
                  M.γ * M.E_next s a (soft_value M Q) := by
    nlinarith
  dsimp [soft_bellman_op]
  have h_recomb := gamma_E_next_delta_sub_C delta C_max s a
  linarith

/-- Functional soft Bellman induction step upper bound (Paper L653-L655):
    If next step soft value satisfies the upper discrepancy bound:
      soft_value(Q)(s') ≤ (V_g*(s') - λ(s) V_o*(s')) + δ(s,s') + B_max(s')
    then T_soft(Q)(s,a) ≤ Q_Σ(s,a) + B_step_op δ B_max s a. -/
theorem soft_bellman_op_upper_step (r_sigma : M.S → M.A → ℝ) (Q_sigma : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ)
    (delta : M.S → M.S → ℝ) (B_max : ValueFunction M) (Q : QFunction M)
    (s : M.S) (a : M.A)
    (h_Q_sigma : Q_sigma s a = r_sigma s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_val : ∀ s', soft_value M Q s' ≤ (Vg' s' - lambda s * Vo' s') + delta s s' + B_max s') :
    soft_bellman_op r_sigma Q s a ≤ Q_sigma s a + B_step_op delta B_max s a := by
  have h_mono : M.E_next s a (soft_value M Q) ≤
                M.E_next s a (fun s' => (Vg' s' - lambda s * Vo' s') + (delta s s' + B_max s')) := by
    apply M.E_next_mono
    intro s'
    have h_le := h_soft_val s'
    linarith
  have h_add : M.E_next s a (fun s' => (Vg' s' - lambda s * Vo' s') + (delta s s' + B_max s')) =
               M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s') +
               M.E_next s a (fun s' => delta s s' + B_max s') := by
    exact M.E_next_add s a (fun s' => Vg' s' - lambda s * Vo' s') (fun s' => delta s s' + B_max s')
  rw [h_add] at h_mono
  have h_gamma_pos := M.hγ_pos
  have h_mul_le : M.γ * M.E_next s a (soft_value M Q) ≤
                  M.γ * (M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s') +
                         M.E_next s a (fun s' => delta s s' + B_max s')) := by
    nlinarith
  dsimp [soft_bellman_op, B_step_op]
  linarith

/-- Lemma 3 (Functional Soft Bellman Induction Step, Paper L604-L650):
    Proves the functional induction step (k ⟹ k+1):
    If Q satisfies the inductive discrepancy bounds relative to Q_Σ with error bounds (B_max, C_max),
    then one step of the soft Bellman operator satisfies:
      Q_Σ(s,a) - C_step_op δ C_max s a ≤ soft_bellman_op r_Σ Q s a ∧
      soft_bellman_op r_Σ Q s a ≤ Q_Σ(s,a) + B_step_op δ B_max s a. -/
theorem soft_bellman_induction_step (r_sigma : M.S → M.A → ℝ) (Q_sigma : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ)
    (delta : M.S → M.S → ℝ) (B_max C_max : ValueFunction M) (Q : QFunction M)
    (s : M.S) (a : M.A)
    (h_Q_sigma : Q_sigma s a = r_sigma s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_lower : ∀ s', soft_value M Q s' ≥ (Vg' s' - lambda s * Vo' s') + delta s s' - C_max s')
    (h_soft_upper : ∀ s', soft_value M Q s' ≤ (Vg' s' - lambda s * Vo' s') + delta s s' + B_max s') :
    Q_sigma s a - C_step_op delta C_max s a ≤ soft_bellman_op r_sigma Q s a ∧
    soft_bellman_op r_sigma Q s a ≤ Q_sigma s a + B_step_op delta B_max s a := by
  have h_low := soft_bellman_op_lower_step r_sigma Q_sigma Vg' Vo' lambda delta C_max Q s a h_Q_sigma h_soft_lower
  have h_up := soft_bellman_op_upper_step r_sigma Q_sigma Vg' Vo' lambda delta B_max Q s a h_Q_sigma h_soft_upper
  exact ⟨h_low, h_up⟩

/-- Lemma 3 (Step Preservation of Q-Bounds):
    At error step bounds B_next = B_step_op δ B_max and C_next = C_step_op δ C_max,
    soft Bellman backup preserves the functional sandwich bounds:
      Q_Σ(s,a) - C_next s a ≤ soft_bellman_op r_Σ Q s a ∧
      soft_bellman_op r_Σ Q s a ≤ Q_Σ(s,a) + B_next s a. -/
theorem soft_bellman_step (r_sigma : M.S → M.A → ℝ) (Q_sigma : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ)
    (delta : M.S → M.S → ℝ) (B_max C_max : ValueFunction M) (Q : QFunction M)
    (B_next C_next : QFunction M)
    (h_B_next : B_next = B_step_op delta B_max)
    (h_C_next : C_next = C_step_op delta C_max)
    (s : M.S) (a : M.A)
    (h_Q_sigma : Q_sigma s a = r_sigma s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_lower : ∀ s', soft_value M Q s' ≥ (Vg' s' - lambda s * Vo' s') + delta s s' - C_max s')
    (h_soft_upper : ∀ s', soft_value M Q s' ≤ (Vg' s' - lambda s * Vo' s') + delta s s' + B_max s') :
    Q_sigma s a - C_next s a ≤ soft_bellman_op r_sigma Q s a ∧
    soft_bellman_op r_sigma Q s a ≤ Q_sigma s a + B_next s a := by
  have h_step := soft_bellman_induction_step r_sigma Q_sigma Vg' Vo' lambda delta B_max C_max Q s a h_Q_sigma h_soft_lower h_soft_upper
  rw [h_B_next, h_C_next]
  exact h_step

/-- k-step soft Bellman iterate starting from Q_init:
    Q^(0) = Q_init, Q^(k+1) = T_soft(Q^(k)) (Paper L600-L604) -/
noncomputable def soft_q_iterate (r : M.S → M.A → ℝ) (Q_init : QFunction M) : ℕ → QFunction M
  | 0 => Q_init
  | k + 1 => soft_bellman_op r (soft_q_iterate r Q_init k)

/-- k-step error bound iterate for upper bound B:
    B^(0) = 0, B^(k+1) = B_step_op δ B_max (Paper L653-L655) -/
def B_iterate (delta : M.S → M.S → ℝ) (B_max : ValueFunction M) : ℕ → QFunction M
  | 0 => fun _ _ => 0
  | _ + 1 => B_step_op delta B_max

/-- k-step error bound iterate for lower bound C:
    C^(0) = 0, C^(k+1) = C_step_op δ C_max (Paper L600-L601, L649) -/
def C_iterate (delta : M.S → M.S → ℝ) (C_max : ValueFunction M) : ℕ → QFunction M
  | 0 => fun _ _ => 0
  | _ + 1 => C_step_op delta C_max

/-- Single-step preservation of inductive bounds under soft Bellman iterate (Paper L604-L655). -/
theorem soft_q_iterate_step (r_sigma : M.S → M.A → ℝ) (Q_sigma Q_init : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ)
    (delta : M.S → M.S → ℝ) (B_max C_max : ValueFunction M)
    (k : ℕ) (s : M.S) (a : M.A)
    (h_Q_sigma : Q_sigma s a = r_sigma s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_lower : ∀ s', soft_value M (soft_q_iterate r_sigma Q_init k) s' ≥ (Vg' s' - lambda s * Vo' s') + delta s s' - C_max s')
    (h_soft_upper : ∀ s', soft_value M (soft_q_iterate r_sigma Q_init k) s' ≤ (Vg' s' - lambda s * Vo' s') + delta s s' + B_max s') :
    Q_sigma s a - C_iterate delta C_max (k + 1) s a ≤ soft_q_iterate r_sigma Q_init (k + 1) s a ∧
    soft_q_iterate r_sigma Q_init (k + 1) s a ≤ Q_sigma s a + B_iterate delta B_max (k + 1) s a := by
  dsimp [soft_q_iterate, B_iterate, C_iterate]
  exact soft_bellman_induction_step r_sigma Q_sigma Vg' Vo' lambda delta B_max C_max
    (soft_q_iterate r_sigma Q_init k) s a h_Q_sigma h_soft_lower h_soft_upper

/-- Bridge lemma: Points-wise Q-bounds imply soft value bounds via monotonicity (Paper L608-L647). -/
lemma soft_value_bounds_of_q_bounds
    (Q Q_sigma : QFunction M) (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ)
    (delta : M.S → M.S → ℝ) (B_max C_max : ValueFunction M)
    (h_soft_Q_sigma : ∀ s s', soft_value M Q_sigma s' = (Vg' s' - lambda s * Vo' s') + delta s s')
    (h_dom_upper : ∀ s' a', Q s' a' - B_max s' ≤ Q_sigma s' a')
    (h_dom_lower : ∀ s' a', Q_sigma s' a' - C_max s' ≤ Q s' a')
    (h_int_Q : ∀ s', Integrable (fun a => Real.exp (Q s' a)) M.μA)
    (h_int_sigma : ∀ s', Integrable (fun a => Real.exp (Q_sigma s' a)) M.μA)
    (h_pos_Q : ∀ s', 0 < ∫ a, Real.exp (Q s' a) ∂M.μA)
    (h_pos_sigma : ∀ s', 0 < ∫ a, Real.exp (Q_sigma s' a) ∂M.μA)
    (s s' : M.S) :
    soft_value M Q s' ≥ (Vg' s' - lambda s * Vo' s') + delta s s' - C_max s' ∧
    soft_value M Q s' ≤ (Vg' s' - lambda s * Vo' s') + delta s s' + B_max s' := by
  have h_up_le := soft_value_mono_sub_max_at M Q_sigma Q s' (B_max s') (h_dom_upper s') (h_int_sigma s') (h_int_Q s') (h_pos_Q s')
  have h_low_le := soft_value_mono_sub_max_at M Q Q_sigma s' (C_max s') (h_dom_lower s') (h_int_Q s') (h_int_sigma s') (h_pos_sigma s')
  rw [h_soft_Q_sigma s s'] at h_up_le h_low_le
  constructor
  · linarith
  · linarith

/-- Lemma 3 (Full Induction on Soft Bellman Iterates, Paper L600-L655):
    For all k ∈ ℕ, the iterate Q^(k+1) satisfies the inductive error bounds:
    Q_Σ(s,a) - C_step_op δ C_max s a ≤ Q^(k+1)(s,a) ≤ Q_Σ(s,a) + B_step_op δ B_max s a. -/
theorem soft_q_iterate_induction (r_sigma : M.S → M.A → ℝ) (Q_sigma : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ) (delta : M.S → M.S → ℝ)
    (B_max C_max : ValueFunction M)
    (h_Q_sigma : ∀ s a, Q_sigma s a = r_sigma s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_Q_sigma : ∀ s s', soft_value M Q_sigma s' = (Vg' s' - lambda s * Vo' s') + delta s s')
    (h_B_bound : ∀ s a, B_step_op delta B_max s a ≤ B_max s)
    (h_C_bound : ∀ s a, C_step_op delta C_max s a ≤ C_max s)
    (h_B_max_nonneg : ∀ s, 0 ≤ B_max s)
    (h_C_max_nonneg : ∀ s, 0 ≤ C_max s)
    (h_int_iter : ∀ k s', Integrable (fun a => Real.exp (soft_q_iterate r_sigma Q_sigma k s' a)) M.μA)
    (h_int_sigma : ∀ s', Integrable (fun a => Real.exp (Q_sigma s' a)) M.μA)
    (h_pos_iter : ∀ k s', 0 < ∫ a, Real.exp (soft_q_iterate r_sigma Q_sigma k s' a) ∂M.μA)
    (h_pos_sigma : ∀ s', 0 < ∫ a, Real.exp (Q_sigma s' a) ∂M.μA)
    (k : ℕ) :
    ∀ s a, Q_sigma s a - C_step_op delta C_max s a ≤ soft_q_iterate r_sigma Q_sigma (k + 1) s a ∧
           soft_q_iterate r_sigma Q_sigma (k + 1) s a ≤ Q_sigma s a + B_step_op delta B_max s a := by
  induction k with
  | zero =>
    intro s a
    dsimp [soft_q_iterate]
    have h_dom_up : ∀ s' a', soft_q_iterate r_sigma Q_sigma 0 s' a' - B_max s' ≤ Q_sigma s' a' := by
      intro s' a'
      dsimp [soft_q_iterate]
      have := h_B_max_nonneg s'
      linarith
    have h_dom_low : ∀ s' a', Q_sigma s' a' - C_max s' ≤ soft_q_iterate r_sigma Q_sigma 0 s' a' := by
      intro s' a'
      dsimp [soft_q_iterate]
      have := h_C_max_nonneg s'
      linarith
    have h_bounds : ∀ s s',
        soft_value M (soft_q_iterate r_sigma Q_sigma 0) s' ≥ (Vg' s' - lambda s * Vo' s') + delta s s' - C_max s' ∧
        soft_value M (soft_q_iterate r_sigma Q_sigma 0) s' ≤ (Vg' s' - lambda s * Vo' s') + delta s s' + B_max s' := by
      intro s s'
      exact soft_value_bounds_of_q_bounds (soft_q_iterate r_sigma Q_sigma 0) Q_sigma Vg' Vo' lambda delta B_max C_max
        h_soft_Q_sigma h_dom_up h_dom_low (h_int_iter 0) h_int_sigma (h_pos_iter 0) h_pos_sigma s s'
    have h_step := soft_bellman_induction_step r_sigma Q_sigma Vg' Vo' lambda delta B_max C_max
      (soft_q_iterate r_sigma Q_sigma 0) s a (h_Q_sigma s a) (fun s' => (h_bounds s s').1) (fun s' => (h_bounds s s').2)
    exact h_step
  | succ n ih =>
    intro s a
    dsimp [soft_q_iterate]
    have h_dom_up : ∀ s' a', soft_q_iterate r_sigma Q_sigma (n + 1) s' a' - B_max s' ≤ Q_sigma s' a' := by
      intro s' a'
      have h_ih_up := (ih s' a').2
      have h_B_le := h_B_bound s' a'
      linarith
    have h_dom_low : ∀ s' a', Q_sigma s' a' - C_max s' ≤ soft_q_iterate r_sigma Q_sigma (n + 1) s' a' := by
      intro s' a'
      have h_ih_low := (ih s' a').1
      have h_C_le := h_C_bound s' a'
      linarith
    have h_bounds : ∀ s s',
        soft_value M (soft_q_iterate r_sigma Q_sigma (n + 1)) s' ≥ (Vg' s' - lambda s * Vo' s') + delta s s' - C_max s' ∧
        soft_value M (soft_q_iterate r_sigma Q_sigma (n + 1)) s' ≤ (Vg' s' - lambda s * Vo' s') + delta s s' + B_max s' := by
      intro s s'
      exact soft_value_bounds_of_q_bounds (soft_q_iterate r_sigma Q_sigma (n + 1)) Q_sigma Vg' Vo' lambda delta B_max C_max
        h_soft_Q_sigma h_dom_up h_dom_low (h_int_iter (n + 1)) h_int_sigma (h_pos_iter (n + 1)) h_pos_sigma s s'
    have h_step := soft_bellman_induction_step r_sigma Q_sigma Vg' Vo' lambda delta B_max C_max
      (soft_q_iterate r_sigma Q_sigma (n + 1)) s a (h_Q_sigma s a) (fun s' => (h_bounds s s').1) (fun s' => (h_bounds s s').2)
    exact h_step

/-- Lemma 3 (Q-Bound Derived from Induction and Limit Preservation, Paper L581-L584, L651-L655 - Resolves Finding C3):
    Derives the optimal compound Q-function sandwich bounds directly from the soft Bellman induction
    and limit convergence, without assuming the conclusion as a hypothesis on Q_comp_star. -/
theorem q_bound_lemma3
    (r_sigma : M.S → M.A → ℝ) (Q_sigma Q_comp_star B_star C_star : QFunction M)
    (Vg' Vo' : ValueFunction M) (lambda : M.S → ℝ) (delta : M.S → M.S → ℝ)
    (B_max_star C_max_star : ValueFunction M)
    (h_Q_sigma : ∀ s a, Q_sigma s a = r_sigma s a + M.γ * M.E_next s a (fun s' => Vg' s' - lambda s * Vo' s'))
    (h_soft_Q_sigma : ∀ s s', soft_value M Q_sigma s' = (Vg' s' - lambda s * Vo' s') + delta s s')
    (h_fp_B : B_star = B_step_op delta B_max_star)
    (h_fp_C : C_star = C_step_op delta C_max_star)
    (h_B_bound : ∀ s a, B_step_op delta B_max_star s a ≤ B_max_star s)
    (h_C_bound : ∀ s a, C_step_op delta C_max_star s a ≤ C_max_star s)
    (h_B_max_nonneg : ∀ s, 0 ≤ B_max_star s)
    (h_C_max_nonneg : ∀ s, 0 ≤ C_max_star s)
    (h_int_iter : ∀ k s', Integrable (fun a => Real.exp (soft_q_iterate r_sigma Q_sigma k s' a)) M.μA)
    (h_int_sigma : ∀ s', Integrable (fun a => Real.exp (Q_sigma s' a)) M.μA)
    (h_pos_iter : ∀ k s', 0 < ∫ a, Real.exp (soft_q_iterate r_sigma Q_sigma k s' a) ∂M.μA)
    (h_pos_sigma : ∀ s', 0 < ∫ a, Real.exp (Q_sigma s' a) ∂M.μA)
    (h_lim_Q : ∀ s a, Filter.Tendsto (fun k => soft_q_iterate r_sigma Q_sigma (k + 1) s a) Filter.atTop (nhds (Q_comp_star s a))) :
    ∀ s a, Q_sigma s a - C_star s a ≤ Q_comp_star s a ∧
           Q_comp_star s a ≤ Q_sigma s a + B_star s a := by
  intro s a
  have h_iter := soft_q_iterate_induction r_sigma Q_sigma Vg' Vo' lambda delta B_max_star C_max_star
    h_Q_sigma h_soft_Q_sigma h_B_bound h_C_bound h_B_max_nonneg h_C_max_nonneg
    h_int_iter h_int_sigma h_pos_iter h_pos_sigma
  have h_low : Q_sigma s a - C_star s a ≤ Q_comp_star s a := by
    have h_bound_k : ∀ k, Q_sigma s a - C_star s a ≤ soft_q_iterate r_sigma Q_sigma (k + 1) s a := by
      intro k
      rw [h_fp_C]
      exact (h_iter k s a).1
    have h_lim_const : Filter.Tendsto (fun _ : ℕ => Q_sigma s a - C_star s a) Filter.atTop (nhds (Q_sigma s a - C_star s a)) :=
      tendsto_const_nhds
    exact le_of_tendsto_of_tendsto' h_lim_const (h_lim_Q s a) h_bound_k
  have h_up : Q_comp_star s a ≤ Q_sigma s a + B_star s a := by
    have h_bound_k : ∀ k, soft_q_iterate r_sigma Q_sigma (k + 1) s a ≤ Q_sigma s a + B_star s a := by
      intro k
      rw [h_fp_B]
      exact (h_iter k s a).2
    have h_lim_const : Filter.Tendsto (fun _ : ℕ => Q_sigma s a + B_star s a) Filter.atTop (nhds (Q_sigma s a + B_star s a)) :=
      tendsto_const_nhds
    exact le_of_tendsto_of_tendsto' (h_lim_Q s a) h_lim_const h_bound_k
  exact ⟨h_low, h_up⟩

/-- Backward compatibility alias for q_bound_lemma3. -/
abbrev q_bound_lemma3_from_induction := @q_bound_lemma3

/-- Lemma 3 Grounded (Resolves Finding U1):
    Directly discharges the abstract hypothesis h_soft_Q_sigma via soft_value_compound_q_eq
    for the compound Q-function Q_Σ = compound_q M Qg Qo lambda. -/
theorem q_bound_lemma3_grounded
    (r_sigma : M.S → M.A → ℝ) (Qg Qo Q_comp_star B_star C_star : QFunction M)
    (lambda : M.S → ℝ)
    (B_max_star C_max_star : ValueFunction M)
    (h_pos_discr : ∀ s', 0 < ∫ a, action_discrepancy_factor Qg Qo lambda s' a ∂M.μA)
    (h_Q_sigma : ∀ s a, compound_q M Qg Qo lambda s a = r_sigma s a + M.γ * M.E_next s a (fun s' => soft_value M Qg s' - lambda s * soft_value M Qo s'))
    (h_fp_B : B_star = B_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) B_max_star)
    (h_fp_C : C_star = C_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) C_max_star)
    (h_B_bound : ∀ s a, B_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) B_max_star s a ≤ B_max_star s)
    (h_C_bound : ∀ s a, C_step_op (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda)) C_max_star s a ≤ C_max_star s)
    (h_B_max_nonneg : ∀ s, 0 ≤ B_max_star s)
    (h_C_max_nonneg : ∀ s, 0 ≤ C_max_star s)
    (h_int_iter : ∀ k s', Integrable (fun a => Real.exp (soft_q_iterate r_sigma (compound_q M Qg Qo lambda) k s' a)) M.μA)
    (h_int_sigma : ∀ s', Integrable (fun a => Real.exp (compound_q M Qg Qo lambda s' a)) M.μA)
    (h_pos_iter : ∀ k s', 0 < ∫ a, Real.exp (soft_q_iterate r_sigma (compound_q M Qg Qo lambda) k s' a) ∂M.μA)
    (h_pos_sigma : ∀ s', 0 < ∫ a, Real.exp (compound_q M Qg Qo lambda s' a) ∂M.μA)
    (h_lim_Q : ∀ s a, Filter.Tendsto (fun k => soft_q_iterate r_sigma (compound_q M Qg Qo lambda) (k + 1) s a) Filter.atTop (nhds (Q_comp_star s a))) :
    ∀ s a, compound_q M Qg Qo lambda s a - C_star s a ≤ Q_comp_star s a ∧
           Q_comp_star s a ≤ compound_q M Qg Qo lambda s a + B_star s a := by
  have h_soft_decomp : ∀ s s', soft_value M (compound_q M Qg Qo lambda) s' =
      (soft_value M Qg s' - lambda s * soft_value M Qo s') +
      discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda) s s' := by
    intro s s'
    exact soft_value_compound_q_eq Qg Qo lambda s s' (h_pos_discr s')
  exact q_bound_lemma3 r_sigma (compound_q M Qg Qo lambda) Q_comp_star B_star C_star
    (soft_value M Qg) (soft_value M Qo) lambda (discrepancy M lambda (soft_value M Qo) (log_action_discrepancy Qg Qo lambda))
    B_max_star C_max_star h_Q_sigma h_soft_decomp h_fp_B h_fp_C h_B_bound h_C_bound h_B_max_nonneg h_C_max_nonneg
    h_int_iter h_int_sigma h_pos_iter h_pos_sigma h_lim_Q

/-- Lemma 3 (Fixed-Point Q-Function Bounding via Limit Preservation, Paper L581-L584, L651-L655):
    As the soft Bellman iterates converge to the optimal compound Q-function Q_comp_star
    and the error step iterates converge to (B_star, C_star), the sandwich bounds
    Q_Σ(s,a) - C*(s,a) ≤ Q_comp_star(s,a) ≤ Q_Σ(s,a) + B*(s,a)
    are derived via order preservation of limits (le_of_tendsto_of_tendsto')
    without assuming the conclusion as a hypothesis on Q_comp_star. -/
theorem q_bound_lemma3_fixed_point
    (r_sigma : M.S → M.A → ℝ) (Q_sigma Q_init Q_comp_star B_star C_star : QFunction M)
    (delta : M.S → M.S → ℝ) (B_max_star C_max_star : ValueFunction M)
    (h_iter_bound : ∀ k s a,
      Q_sigma s a - C_iterate delta C_max_star k s a ≤ soft_q_iterate r_sigma Q_init k s a ∧
      soft_q_iterate r_sigma Q_init k s a ≤ Q_sigma s a + B_iterate delta B_max_star k s a)
    (h_lim_Q : ∀ s a, Filter.Tendsto (fun k => soft_q_iterate r_sigma Q_init k s a) Filter.atTop (nhds (Q_comp_star s a)))
    (h_lim_B : ∀ s a, Filter.Tendsto (fun k => B_iterate delta B_max_star k s a) Filter.atTop (nhds (B_star s a)))
    (h_lim_C : ∀ s a, Filter.Tendsto (fun k => C_iterate delta C_max_star k s a) Filter.atTop (nhds (C_star s a))) :
    ∀ s a, Q_sigma s a - C_star s a ≤ Q_comp_star s a ∧
           Q_comp_star s a ≤ Q_sigma s a + B_star s a := by
  intro s a
  have h_low : Q_sigma s a - C_star s a ≤ Q_comp_star s a := by
    have h_bound_k : ∀ k, Q_sigma s a - C_iterate delta C_max_star k s a ≤ soft_q_iterate r_sigma Q_init k s a :=
      fun k => (h_iter_bound k s a).1
    have h_lim_lhs : Filter.Tendsto (fun k => Q_sigma s a - C_iterate delta C_max_star k s a) Filter.atTop (nhds (Q_sigma s a - C_star s a)) :=
      tendsto_const_nhds.sub (h_lim_C s a)
    exact le_of_tendsto_of_tendsto' h_lim_lhs (h_lim_Q s a) h_bound_k
  have h_up : Q_comp_star s a ≤ Q_sigma s a + B_star s a := by
    have h_bound_k : ∀ k, soft_q_iterate r_sigma Q_init k s a ≤ Q_sigma s a + B_iterate delta B_max_star k s a :=
      fun k => (h_iter_bound k s a).2
    have h_lim_rhs : Filter.Tendsto (fun k => Q_sigma s a + B_iterate delta B_max_star k s a) Filter.atTop (nhds (Q_sigma s a + B_star s a)) :=
      tendsto_const_nhds.add (h_lim_B s a)
    exact le_of_tendsto_of_tendsto' (h_lim_Q s a) h_lim_rhs h_bound_k
  exact ⟨h_low, h_up⟩

end LeanProofs.Proofs

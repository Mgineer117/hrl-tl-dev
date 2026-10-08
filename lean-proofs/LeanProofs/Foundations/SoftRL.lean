import Mathlib
import LeanProofs.Foundations.MDP

set_option linter.style.header false
set_option linter.style.longLine false

namespace LeanProofs.Foundations

open MeasureTheory

variable (M : MDP)

/-- Soft state-value function V^*(s) = log ∫_A exp(Q^*(s,a')) dμ(a'). -/
noncomputable def soft_value (Q : QFunction M) (s : M.S) : ℝ :=
  Real.log (∫ a, Real.exp (Q s a) ∂M.μA)

/-- MaxEnt Boltzmann policy: π^*(a|s) = exp(Q^*(s,a) - V^*(s)). -/
noncomputable def boltzmann_policy (Q : QFunction M) (s : M.S) (a : M.A) : ℝ :=
  Real.exp (Q s a - soft_value M Q s)

/-- Key Boltzmann policy decomposition identity: log π^*(a|s) = Q^*(s,a) - V^*(s). -/
theorem boltzmann_policy_log_eq (Q : QFunction M) (s : M.S) (a : M.A) :
    Real.log (boltzmann_policy M Q s a) = Q s a - soft_value M Q s := by
  dsimp [boltzmann_policy]
  exact Real.log_exp (Q s a - soft_value M Q s)

/-- Q*(s,a) = log π*(a|s) + V*(s). -/
theorem Q_eq_log_policy_add_V (Q : QFunction M) (s : M.S) (a : M.A) :
    Q s a = Real.log (boltzmann_policy M Q s a) + soft_value M Q s := by
  have h := boltzmann_policy_log_eq M Q s a
  linarith

/-- Compound Q-function Q_Σ(s,a) = Q_g^*(s,a) - λ(s) * Q_o^*(s,a). -/
def compound_q (Qg Qo : QFunction M) (lambda : M.S → ℝ) : QFunction M :=
  fun s a => Qg s a - lambda s * Qo s a

/-- Discrepancy term δ(s, s') from Lemma 3 (Paper Eqn 607):
    δ(s, s') = (λ(s) - λ(s')) V_o^*(s') + log E_{a ~ π_g}[ (π_o^*(a|s'))^{-λ(s')} ].
    Note: The log-expectation term log E_{a ~ π_g}[ (π_o^*(a|s'))^{-λ(s')} ] is abstracted
    as an arbitrary state-dependent function `log_expect_term : M.S → ℝ` since the subsequent
    bounding proofs depend purely on its algebraic position rather than its internal measure-theoretic definition. -/
def discrepancy (lambda : M.S → ℝ) (Vo : ValueFunction M)
    (log_expect_term : M.S → ℝ) (s s' : M.S) : ℝ :=
  (lambda s - lambda s') * Vo s' + log_expect_term s'

/-- Extraction of constant shift from soft value integral:
    log ∫_A exp(f(a) - C) dμ = log ∫_A exp(f(a)) dμ - C. -/
theorem log_integral_sub_const (f : M.A → ℝ) (C : ℝ)
    (hf_pos : 0 < ∫ a, Real.exp (f a) ∂M.μA) :
    Real.log (∫ a, Real.exp (f a - C) ∂M.μA) = Real.log (∫ a, Real.exp (f a) ∂M.μA) - C := by
  have h_exp : ∀ a, Real.exp (f a - C) = Real.exp (-C) * Real.exp (f a) := by
    intro a
    rw [sub_eq_add_neg, Real.exp_add, mul_comm]
  have h_int_eq : (∫ a, Real.exp (f a - C) ∂M.μA) = Real.exp (-C) * (∫ a, Real.exp (f a) ∂M.μA) := by
    simp_rw [h_exp]
    exact integral_const_mul (Real.exp (-C)) (fun a => Real.exp (f a))
  rw [h_int_eq]
  have h_exp_neg_pos : 0 < Real.exp (-C) := Real.exp_pos _
  rw [Real.log_mul (ne_of_gt h_exp_neg_pos) (ne_of_gt hf_pos)]
  rw [Real.log_exp]
  ring

/-- Positivity of the soft partition function when measure is non-trivial and exp(Q) is integrable. -/
lemma soft_partition_pos [NeZero M.μA] (Q : QFunction M) (s : M.S)
    (h_int : Integrable (fun a => Real.exp (Q s a)) M.μA) :
    0 < ∫ a, Real.exp (Q s a) ∂M.μA := by
  have h_nonneg : 0 ≤ᵐ[M.μA] (fun a => Real.exp (Q s a)) :=
    ae_of_all _ (fun a => (Real.exp_pos (Q s a)).le)
  have h_supp : Function.support (fun a => Real.exp (Q s a)) = Set.univ := by
    ext a
    simp [Function.mem_support, Real.exp_ne_zero]
  have h_pos_meas : 0 < M.μA Set.univ := by
    rw [pos_iff_ne_zero]
    exact Measure.measure_univ_ne_zero.mpr (NeZero.ne M.μA)
  rw [integral_pos_iff_support_of_nonneg_ae h_nonneg h_int, h_supp]
  exact h_pos_meas

/-- Pointwise (at state s) monotonicity of soft state-value function under constant shift:
    If Q₂(s, a) - B ≤ Q₁(s, a) for all a, then
    soft_value(Q₂) s - B ≤ soft_value(Q₁) s. -/
theorem soft_value_mono_sub_max_at (Q₁ Q₂ : QFunction M) (s : M.S) (B : ℝ)
    (h_dom : ∀ a, Q₂ s a - B ≤ Q₁ s a)
    (h_int₁ : Integrable (fun a => Real.exp (Q₁ s a)) M.μA)
    (h_int₂ : Integrable (fun a => Real.exp (Q₂ s a)) M.μA)
    (h_pos₂ : 0 < ∫ a, Real.exp (Q₂ s a) ∂M.μA) :
    soft_value M Q₂ s - B ≤ soft_value M Q₁ s := by
  have h_exp_neg : 0 < Real.exp (-B) := Real.exp_pos (-B)
  have h_exp_shift : (fun a => Real.exp (Q₂ s a - B)) = (fun a => Real.exp (-B) * Real.exp (Q₂ s a)) := by
    ext a
    rw [sub_eq_add_neg, Real.exp_add, mul_comm]
  have h_int_shifted : Integrable (fun a => Real.exp (Q₂ s a - B)) M.μA := by
    rw [h_exp_shift]
    exact h_int₂.const_mul (Real.exp (-B))
  have h_integral_factor : (∫ a, Real.exp (Q₂ s a - B) ∂M.μA) = Real.exp (-B) * (∫ a, Real.exp (Q₂ s a) ∂M.μA) := by
    rw [h_exp_shift]
    exact integral_const_mul (Real.exp (-B)) (fun a => Real.exp (Q₂ s a))
  have h_pos_shifted : 0 < ∫ a, Real.exp (Q₂ s a - B) ∂M.μA := by
    rw [h_integral_factor]
    exact mul_pos h_exp_neg h_pos₂
  have h_pointwise : ∀ a, Real.exp (Q₂ s a - B) ≤ Real.exp (Q₁ s a) := by
    intro a
    exact Real.exp_le_exp.mpr (h_dom a)
  have h_integral_mono : (∫ a, Real.exp (Q₂ s a - B) ∂M.μA) ≤ (∫ a, Real.exp (Q₁ s a) ∂M.μA) :=
    integral_mono h_int_shifted h_int₁ h_pointwise
  have h_log_mono : Real.log (∫ a, Real.exp (Q₂ s a - B) ∂M.μA) ≤ Real.log (∫ a, Real.exp (Q₁ s a) ∂M.μA) :=
    Real.log_le_log h_pos_shifted h_integral_mono
  have h_log_shift : Real.log (∫ a, Real.exp (Q₂ s a - B) ∂M.μA) = soft_value M Q₂ s - B := by
    dsimp [soft_value]
    rw [h_integral_factor]
    rw [Real.log_mul (ne_of_gt h_exp_neg) (ne_of_gt h_pos₂)]
    rw [Real.log_exp]
    ring
  rw [h_log_shift] at h_log_mono
  exact h_log_mono

/-- Monotonicity of soft state-value function under uniform pointwise shift (Requirement R1, Paper §4.2):
    If Q₂(s, a) - B ≤ Q₁(s, a) for all s, a, then
    soft_value(Q₂) - B ≤ soft_value(Q₁). -/
theorem soft_value_mono_sub_max (Q₁ Q₂ : QFunction M) (B : ℝ)
    (h_dom : ∀ s a, Q₂ s a - B ≤ Q₁ s a) (s : M.S)
    (h_int₁ : Integrable (fun a => Real.exp (Q₁ s a)) M.μA)
    (h_int₂ : Integrable (fun a => Real.exp (Q₂ s a)) M.μA)
    (h_pos₂ : 0 < ∫ a, Real.exp (Q₂ s a) ∂M.μA) :
    soft_value M Q₂ s - B ≤ soft_value M Q₁ s :=
  soft_value_mono_sub_max_at M Q₁ Q₂ s B (h_dom s) h_int₁ h_int₂ h_pos₂

/-- Variant of soft_value_mono_sub_max using [NeZero M.μA] to automatically discharge partition positivity. -/
theorem soft_value_mono_sub_max' [NeZero M.μA] (Q₁ Q₂ : QFunction M) (B : ℝ)
    (h_dom : ∀ s a, Q₂ s a - B ≤ Q₁ s a) (s : M.S)
    (h_int₁ : Integrable (fun a => Real.exp (Q₁ s a)) M.μA)
    (h_int₂ : Integrable (fun a => Real.exp (Q₂ s a)) M.μA) :
    soft_value M Q₂ s - B ≤ soft_value M Q₁ s :=
  soft_value_mono_sub_max M Q₁ Q₂ B h_dom s h_int₁ h_int₂ (soft_partition_pos M Q₂ s h_int₂)

/-- Variant of soft_value_mono_sub_max stating the inequality in the direction soft_value(Q₁) ≥ soft_value(Q₂) - B. -/
theorem soft_value_mono_sub_max_ge (Q₁ Q₂ : QFunction M) (B : ℝ)
    (h_dom : ∀ s a, Q₁ s a ≥ Q₂ s a - B) (s : M.S)
    (h_int₁ : Integrable (fun a => Real.exp (Q₁ s a)) M.μA)
    (h_int₂ : Integrable (fun a => Real.exp (Q₂ s a)) M.μA)
    (h_pos₂ : 0 < ∫ a, Real.exp (Q₂ s a) ∂M.μA) :
    soft_value M Q₁ s ≥ soft_value M Q₂ s - B := by
  have h_le : ∀ s a, Q₂ s a - B ≤ Q₁ s a := fun s a => h_dom s a
  exact soft_value_mono_sub_max M Q₁ Q₂ B h_le s h_int₁ h_int₂ h_pos₂

end LeanProofs.Foundations

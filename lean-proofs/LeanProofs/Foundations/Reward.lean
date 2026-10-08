import Mathlib
import LeanProofs.Foundations.MDP

set_option linter.style.header false
set_option linter.style.longLine false

namespace LeanProofs.Foundations

/-- Composition constants ξ = ⟨L, k, ε⟩ for Contrastive Policy Composition. -/
structure CompositionConstants where
  /-- Maximum repulsion gain L > 0 -/
  L : ℝ
  hL_pos : 0 < L
  /-- Transition steepness parameter k > 0 -/
  k : ℝ
  hk_pos : 0 < k
  /-- Safety margin ε ≥ 0 -/
  ε : ℝ
  hε_nonneg : 0 ≤ ε

/-- Repulsion gain function λ(s): logistic sigmoid scaling monotonically to L as robustness drops.
    Matches Paper Eqn (5) / L384: λ(s) = L / (1 + exp(k * (ρ(s, ¬ψ_o) - ε))). -/
noncomputable def repulsion_gain (ξ : CompositionConstants) (rho_safe : ℝ) : ℝ :=
  ξ.L / (1 + Real.exp (ξ.k * (rho_safe - ξ.ε)))

lemma repulsion_gain_pos (ξ : CompositionConstants) (rho_safe : ℝ) :
    0 < repulsion_gain ξ rho_safe := by
  dsimp [repulsion_gain]
  apply div_pos ξ.hL_pos
  have : 0 < Real.exp (ξ.k * (rho_safe - ξ.ε)) := Real.exp_pos _
  linarith

lemma repulsion_gain_le_L (ξ : CompositionConstants) (rho_safe : ℝ) :
    repulsion_gain ξ rho_safe ≤ ξ.L := by
  dsimp [repulsion_gain]
  have h_denom : 1 ≤ 1 + Real.exp (ξ.k * (rho_safe - ξ.ε)) := by
    have : 0 < Real.exp (ξ.k * (rho_safe - ξ.ε)) := Real.exp_pos _
    linarith
  have h_denom_pos : 0 < 1 + Real.exp (ξ.k * (rho_safe - ξ.ε)) := by linarith
  exact div_le_self ξ.hL_pos.le h_denom

/-- When safe robustness is non-positive (state approaches or enters obstacle),
    the repulsion gain exceeds L / 2 (Paper Lemma 1, Supplementary Eq. 60). -/
lemma repulsion_gain_ge_half_L (ξ : CompositionConstants) (rho_safe : ℝ)
    (hrho_le : rho_safe ≤ 0) :
    repulsion_gain ξ rho_safe ≥ ξ.L / 2 := by
  dsimp [repulsion_gain]
  have h_diff : rho_safe - ξ.ε ≤ 0 := by linarith [ξ.hε_nonneg]
  have h_inner : ξ.k * (rho_safe - ξ.ε) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos ξ.hk_pos.le h_diff
  have h_exp_le_one : Real.exp (ξ.k * (rho_safe - ξ.ε)) ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr h_inner
  have h_denom_le_two : 1 + Real.exp (ξ.k * (rho_safe - ξ.ε)) ≤ 2 := by linarith
  have h_denom_pos : 0 < 1 + Real.exp (ξ.k * (rho_safe - ξ.ε)) := by
    have : 0 < Real.exp (ξ.k * (rho_safe - ξ.ε)) := Real.exp_pos _
    linarith
  have h_div : ξ.L / 2 ≤ ξ.L / (1 + Real.exp (ξ.k * (rho_safe - ξ.ε))) :=
    div_le_div_of_nonneg_left ξ.hL_pos.le h_denom_pos h_denom_le_two
  exact h_div

/-- Boundary obstacle penalty lower bound:
    If obstacle penetration displacement r_o ≥ delta_step > 0 and λ ≥ L / 2,
    then λ * r_o ≥ (L / 2) * delta_step (Paper Lemma 1, Eq. 64). -/
lemma boundary_obstacle_penalty_ge (lambda r_o L delta_step : ℝ)
    (h_lambda : lambda ≥ L / 2)
    (h_ro : r_o ≥ delta_step)
    (hdelta_nonneg : 0 ≤ delta_step)
    (hL_nonneg : 0 ≤ L) :
    lambda * r_o ≥ (L / 2) * delta_step := by
  have h_half_L : 0 ≤ L / 2 := by linarith
  have h_lambda_nonneg : 0 ≤ lambda := le_trans h_half_L h_lambda
  nlinarith

/-- Compound reward for CPC: r_Σ(s,a) = r_g(s,a) - λ(s) * r_o(s,a). -/
def compound_reward (rg ro : ℝ) (lambda : ℝ) : ℝ :=
  rg - lambda * ro

/-- Compound reward obstacle upper bound (Resolves Finding U2):
    When goal reward is bounded by R_max and obstacle penalty is at least Lambda_min,
    the compound reward is at most R_max - Lambda_min. -/
lemma compound_reward_obstacle_bound (rg ro lambda R_max Lambda_min : ℝ)
    (h_rg : rg ≤ R_max) (h_pen : lambda * ro ≥ Lambda_min) :
    compound_reward rg ro lambda ≤ R_max - Lambda_min := by
  dsimp [compound_reward]
  linarith

/-- Compound reward safe lower bound:
    When goal reward is at least R_min and obstacle penalty is 0,
    the compound reward is at least R_min. -/
lemma compound_reward_safe_lower_bound (rg ro lambda R_min : ℝ)
    (h_rg : rg ≥ R_min) (h_pen : lambda * ro ≤ 0) :
    compound_reward rg ro lambda ≥ R_min := by
  dsimp [compound_reward]
  linarith

end LeanProofs.Foundations

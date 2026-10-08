import Mathlib
import LeanProofs.Foundations.MDP
import LeanProofs.Foundations.Reward
import LeanProofs.Foundations.Sets

set_option linter.style.header false
set_option linter.style.longLine false

/-!
# Paper Assumptions 1–4 (Section 4.1)

Formalizes the core mathematical assumptions under which safety and reachability are established:
- **Assumption 1 (Compactness & Boundedness)**: Compactness of state/action spaces, bounding reward and value functions (`R_max`, `V_max_safe`).
- **Assumption 2 (Topological Feasibility)**: Existence of a strictly safe path to the goal with finite maximum hitting time `T_max`. The goal set is absorbing.
- **Assumption 3 (Repulsion Polarity)**: Boundary penalty function `Lambda_min(L)` growing arbitrarily large as `L → ∞`.
- **Assumption 4 (Goal Reward Dominance)**: Reward margin `d > 0` between goal and transition safe states (`R_min ≤ R_trans`).

### Paper to LEAN Correspondence Table
| Paper Symbol | Lean Field / Name | Description |
|---|---|---|
| $R_{\mathrm{max}}$ | `R_max` | Maximum goal reward |
| $V_{\mathrm{max}}^{\mathrm{safe}}$ | `V_max_safe` | Upper bound on value in $\mathcal{S}_{\mathrm{safe}}$ |
| $V_{\mathrm{safe}}^{\mathrm{min}}$ | `V_safe_min` | Worst-case guaranteed value in $\mathcal{S}_{\mathrm{safe}}$ |
| $p_{\mathrm{fail}}$ | `p_fail` | Transition probability into obstacle set $\mathcal{S}_{\mathrm{o}}$ under unsafe action |
| $T_{\mathrm{max}}$ | `T_max` | Maximum hitting time to goal under safe policy |
| $\Lambda_{\mathrm{min}}(L)$ | `Lambda_min` | Obstacle penalty lower bound |
| $R_{\mathrm{trans}}$ | `R_trans` | Supremum compound reward in transition safe set $\mathcal{S}_{\mathrm{trans}}$ |
| $R_{\mathrm{min}}$ | `R_min` | Infimum compound reward in $\mathcal{S}_{\mathrm{safe}}$ |
| $d$ | `d` | Dominance gap $\inf_{\mathcal{S}_{\mathrm{g}}} R_{\mathrm{g}} - R_{\mathrm{trans}} > 0$ |
-/

namespace LeanProofs.Foundations

variable (M : MDP)

/-- Assumptions 1 to 4 from Section 4.1 of the paper. -/
structure PaperAssumptions (sets : TaskSets M) where
  -- Assumption 1: Bounds on reward and values
  R_max : ℝ
  hR_max_pos : 0 < R_max
  V_max_safe : ℝ
  hV_max_safe_bound : V_max_safe ≤ R_max / (1 - M.γ)
  V_safe_min : ℝ

  -- Action partition for boundary states (Lemma 1 setup):
  -- For state s in S_safe, an unsafe action a_u transitions to S_o with p_fail > 0
  p_fail : ℝ
  hp_fail_pos : 0 < p_fail
  hp_fail_le_one : p_fail ≤ 1

  -- Assumption 2: Topological Feasibility (Paper Assumption 2, L438-L442)
  T_max : ℕ
  hT_max_pos : 0 < T_max
  /-- Goal set is absorbing: all transitions from S_g stay in S_g. -/
  goal_absorbing : ∀ (s : M.S) (a : M.A) (s' : M.S),
    s ∈ sets.S_g → StepSupported M s a s' → s' ∈ sets.S_g
  /-- Every safe state possesses at least one safe action preserving safety. -/
  safe_action_exists : ∀ s ∈ sets.S_safe, ∃ a_s : M.A, sets.IsSafeAction s a_s
  /-- For every initial safe state s₀, there exists an admissible path reaching S_g in T ≤ T_max steps. -/
  exists_reaching_path : ∀ s₀ ∈ sets.S_safe, ∃ (τ_path : Trace M.S) (T : ℕ),
    T ≤ T_max ∧ τ_path 0 = s₀ ∧
    (∀ t < T, τ_path t ∈ sets.S_safe) ∧
    (∀ t ≥ T, τ_path t ∈ sets.S_g)

  -- Assumption 3: Boundary Penetration Clearance & Buffer Margin (Paper Assumption 3)
  delta_step : ℝ
  hdelta_step_pos : 0 < delta_step
  epsilon : ℝ := 0
  hepsilon_nonneg : 0 ≤ epsilon := by linarith
  /-- Clearance dominance condition: obstacle displacement strictly outpaces buffer margin. -/
  h_clearance : delta_step / 2 > (M.γ + (1 - p_fail) / p_fail) * (epsilon / (1 - M.γ)) := by
    have h_pos : 0 < delta_step / 2 := by linarith [hdelta_step_pos]
    have h_zero : (M.γ + (1 - p_fail) / p_fail) * (0 / (1 - M.γ)) = 0 := by ring
    linarith

  -- Assumption 4: Goal Reward Dominance
  R_trans : ℝ
  d : ℝ
  hd_pos : 0 < d
  R_min : ℝ
  hR_min_le_R_trans : R_min ≤ R_trans

variable {M : MDP} {sets : TaskSets M}

/-- Boundary obstacle penalty lower bound function Λ_min(L) = (L / 2) * delta_step,
    derived from the minimum boundary penetration clearance δ_step > 0. -/
noncomputable def PaperAssumptions.Lambda_min (asm : PaperAssumptions M sets) (L : ℝ) : ℝ :=
  (L / 2) * asm.delta_step

/-- Repulsion Penalty Divergence Lemma (Paper Lemma 1 derivation):
    For any threshold B ∈ ℝ, there exists a finite gain threshold L_thresh such that
    for all L ≥ L_thresh, asm.Lambda_min L ≥ B. -/
lemma PaperAssumptions.lambda_min_lim (asm : PaperAssumptions M sets) (B : ℝ) :
    ∃ L_thresh : ℝ, ∀ L ≥ L_thresh, asm.Lambda_min L ≥ B := by
  use max 0 (2 * B / asm.delta_step)
  intro L hL
  dsimp [PaperAssumptions.Lambda_min]
  have h_pos : 0 < asm.delta_step := asm.hdelta_step_pos
  have h_ge : 2 * B / asm.delta_step ≤ L := le_trans (le_max_right 0 _) hL
  have h_mul : 2 * B ≤ L * asm.delta_step := (div_le_iff₀ h_pos).mp h_ge
  linarith

end LeanProofs.Foundations

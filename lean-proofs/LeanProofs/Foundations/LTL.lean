import Mathlib
import LeanProofs.Foundations.MDP
import LeanProofs.Foundations.Sets
import LeanProofs.Foundations.Assumptions

set_option linter.style.header false
set_option linter.style.longLine false

namespace LeanProofs.Foundations


/-- Temporal modality: globally (always) `P` holds along trace `τ`. -/
def globally {S : Type*} (P : S → Prop) (τ : Trace S) : Prop :=
  ∀ t : ℕ, P (τ t)

/-- Temporal modality: eventually (in finite time) `P` holds along trace `τ`. -/
def eventually {S : Type*} (P : S → Prop) (τ : Trace S) : Prop :=
  ∃ t : ℕ, P (τ t)

/-- Reach-avoid specification: `F goal ∧ G safe`. -/
def reach_avoid {S : Type*} (goal safe : S → Prop) (τ : Trace S) : Prop :=
  eventually goal τ ∧ globally safe τ

/-- Conjunction lemma connecting eventual reachability and global safety to reach-avoid. -/
theorem reach_avoid_intro {S : Type*} {goal safe : S → Prop} {τ : Trace S}
    (h_safe : globally safe τ) (h_goal : eventually goal τ) :
    reach_avoid goal safe τ :=
  ⟨h_goal, h_safe⟩

/-- A trace τ is induced by policy π from state s₀ if τ(0) = s₀ and
    each successive state τ(t+1) is supported by the transition distribution under π(τ(t)). -/
def InducesTrajectory (M : MDP) (π : M.S → M.A) (s₀ : M.S) (τ : Trace M.S) : Prop :=
  τ 0 = s₀ ∧ ∀ t : ℕ, StepSupported M (τ t) (π (τ t)) (τ (t + 1))

/-- Cumulative discounted return of a trajectory under reward r. -/
noncomputable def trajectory_return (M : MDP) (r : M.S → ℝ) (τ : Trace M.S) : ℝ :=
  ∑' t : ℕ, M.γ ^ t * r (τ t)

/-- Bellman optimality guarantee: an optimal trajectory achieves the optimal state value from initial state s₀. -/
def IsOptimalTrajectory (M : MDP) (r : M.S → ℝ) (V_star : ValueFunction M) (τ : Trace M.S) : Prop :=
  trajectory_return M r τ = V_star (τ 0)

/-- State-specialized optimal trajectory predicate connecting initial state s₀, value function, and trajectory return. -/
def IsOptimalTrajectoryFrom (M : MDP) (s₀ : M.S) (r : M.S → ℝ) (V_star : ValueFunction M) (τ : Trace M.S) : Prop :=
  τ 0 = s₀ ∧ IsOptimalTrajectory M r V_star τ

/-- Telescoping finite sum formula for real sequences: ∑_{i=0}^{n-1} (f(i) - f(i+1)) = f(0) - f(n). -/
theorem telescope_sum (f : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range n, (f i - f (i + 1)) = f 0 - f n := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, ih]
    ring

/-- Bellman transversality derivation (Option A):
    Any execution trace τ along which V*(τ t) satisfies the Bellman relation
    V*(τ t) = r(τ t) + γ V*(τ (t+1)) with bounded value function achieves
    trajectory_return M r τ = V*(τ 0), rigorously deriving IsOptimalTrajectory. -/
theorem optimal_trace_is_optimal_trajectory (M : MDP) (r : M.S → ℝ) (V_star : ValueFunction M)
    (τ : Trace M.S)
    (h_bellman : ∀ t : ℕ, V_star (τ t) = r (τ t) + M.γ * V_star (τ (t + 1)))
    (B : ℝ) (h_bound : ∀ s, |V_star s| ≤ B) :
    IsOptimalTrajectory M r V_star τ := by
  have hγ_nonneg : 0 ≤ M.γ := le_of_lt M.hγ_pos
  have hγ_lt_one : M.γ < 1 := M.hγ_lt_one
  -- Bound on r(τ t)
  have hr_bound : ∀ t : ℕ, |r (τ t)| ≤ (1 + M.γ) * B := by
    intro t
    have h_b := h_bellman t
    have h_r : r (τ t) = V_star (τ t) - M.γ * V_star (τ (t + 1)) := by linarith
    rw [h_r]
    have h_tri := abs_sub (V_star (τ t)) (M.γ * V_star (τ (t + 1)))
    have h_mul : |M.γ * V_star (τ (t + 1))| = M.γ * |V_star (τ (t + 1))| := by
      rw [abs_mul, abs_of_nonneg hγ_nonneg]
    rw [h_mul] at h_tri
    have h1 := h_bound (τ t)
    have h2 := h_bound (τ (t + 1))
    have h_le : |V_star (τ t)| + M.γ * |V_star (τ (t + 1))| ≤ B + M.γ * B := by
      nlinarith
    have h_ring : B + M.γ * B = (1 + M.γ) * B := by ring
    rw [h_ring] at h_le
    exact le_trans h_tri h_le
  -- Summability of M.γ ^ t * r(τ t)
  have h_geom := summable_geometric_of_lt_one hγ_nonneg hγ_lt_one
  have h_bound_sum : Summable (fun t : ℕ => M.γ ^ t * ((1 + M.γ) * B)) :=
    h_geom.mul_right ((1 + M.γ) * B)
  have h_norm_le : ∀ t : ℕ, ‖M.γ ^ t * r (τ t)‖ ≤ M.γ ^ t * ((1 + M.γ) * B) := by
    intro t
    rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_nonneg hγ_nonneg]
    exact mul_le_mul_of_nonneg_left (hr_bound t) (pow_nonneg hγ_nonneg t)
  have h_summable : Summable (fun t : ℕ => M.γ ^ t * r (τ t)) :=
    Summable.of_norm_bounded h_bound_sum h_norm_le
  -- Finite partial sums
  have h_step : ∀ t : ℕ, M.γ ^ t * r (τ t) =
      (fun i => M.γ ^ i * V_star (τ i)) t - (fun i => M.γ ^ i * V_star (τ i)) (t + 1) := by
    intro t
    have h_b := h_bellman t
    have h_r : r (τ t) = V_star (τ t) - M.γ * V_star (τ (t + 1)) := by linarith
    dsimp
    rw [h_r]
    have h_pow : M.γ ^ (t + 1) = M.γ ^ t * M.γ := pow_succ M.γ t
    rw [h_pow]
    ring
  have h_partial : ∀ n : ℕ, ∑ t ∈ Finset.range n, M.γ ^ t * r (τ t) =
      V_star (τ 0) - M.γ ^ n * V_star (τ n) := by
    intro n
    have h_sum : ∑ t ∈ Finset.range n, M.γ ^ t * r (τ t) =
        ∑ t ∈ Finset.range n, ((fun i => M.γ ^ i * V_star (τ i)) t - (fun i => M.γ ^ i * V_star (τ i)) (t + 1)) := by
      apply Finset.sum_congr rfl
      intro x _
      exact h_step x
    rw [h_sum, telescope_sum]
    simp only [pow_zero, one_mul]
  -- Limit of M.γ ^ n * V_star (τ n) is 0
  have h_lim_pow : Filter.Tendsto (fun n : ℕ => M.γ ^ n) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hγ_nonneg hγ_lt_one
  have h_lim_mul_B : Filter.Tendsto (fun n : ℕ => M.γ ^ n * B) Filter.atTop (nhds 0) := by
    have h_mul := Filter.Tendsto.mul_const B h_lim_pow
    rw [MulZeroClass.zero_mul] at h_mul
    exact h_mul
  have h_lim_neg_mul_B : Filter.Tendsto (fun n : ℕ => - (M.γ ^ n * B)) Filter.atTop (nhds 0) := by
    have h_neg := h_lim_mul_B.neg
    rw [neg_zero] at h_neg
    exact h_neg
  have h_lim_rem : Filter.Tendsto (fun n : ℕ => M.γ ^ n * V_star (τ n)) Filter.atTop (nhds 0) := by
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' h_lim_neg_mul_B h_lim_mul_B
    · exact Filter.Eventually.of_forall (fun n => by
        have h_abs := h_bound (τ n)
        rw [abs_le] at h_abs
        have h_pow_pos : 0 ≤ M.γ ^ n := pow_nonneg hγ_nonneg n
        nlinarith)
    · exact Filter.Eventually.of_forall (fun n => by
        have h_abs := h_bound (τ n)
        rw [abs_le] at h_abs
        have h_pow_pos : 0 ≤ M.γ ^ n := pow_nonneg hγ_nonneg n
        nlinarith)
  -- Partial sums tend to V_star (τ 0)
  have h_tendsto : Filter.Tendsto (fun n : ℕ => ∑ t ∈ Finset.range n, M.γ ^ t * r (τ t)) Filter.atTop (nhds (V_star (τ 0))) := by
    have h_eq : (fun n : ℕ => ∑ t ∈ Finset.range n, M.γ ^ t * r (τ t)) = (fun n => V_star (τ 0) - M.γ ^ n * V_star (τ n)) := by
      funext n
      exact h_partial n
    rw [h_eq]
    have h_sub : Filter.Tendsto (fun n : ℕ => V_star (τ 0) - M.γ ^ n * V_star (τ n)) Filter.atTop (nhds (V_star (τ 0) - 0)) :=
      tendsto_const_nhds.sub h_lim_rem
    rw [sub_zero] at h_sub
    exact h_sub
  -- Since series is summable, its tsum is the limit of its partial sums
  have h_sum_lim := h_summable.hasSum.tendsto_sum_nat
  have h_unique := tendsto_nhds_unique h_sum_lim h_tendsto
  dsimp [IsOptimalTrajectory, trajectory_return]
  exact h_unique


/-- Discounted trajectory return under constant reward evaluates to the geometric series sum C / (1 - γ). -/
theorem trajectory_return_const (M : MDP) (C : ℝ) (τ : Trace M.S) :
    trajectory_return M (fun _ => C) τ = C / (1 - M.γ) := by
  dsimp [trajectory_return]
  rw [tsum_mul_right]
  rw [tsum_geometric_of_lt_one M.hγ_pos.le M.hγ_lt_one]
  ring

/-- Discounted trajectory return under constant reward with flipped argument order (τ, C). -/
theorem trajectory_return_const' (M : MDP) (τ : Trace M.S) (C : ℝ) :
    trajectory_return M (fun _ => C) τ = C / (1 - M.γ) :=
  trajectory_return_const M C τ

/-- Monotonicity of trajectory returns for term-wise ordered reward sequences. -/
theorem trajectory_return_mono (M : MDP) (r₁ r₂ : M.S → ℝ) (τ : Trace M.S)
    (h_le : ∀ t : ℕ, r₁ (τ t) ≤ r₂ (τ t))
    (h_sum₁ : Summable (fun t : ℕ => M.γ ^ t * r₁ (τ t)))
    (h_sum₂ : Summable (fun t : ℕ => M.γ ^ t * r₂ (τ t))) :
    trajectory_return M r₁ τ ≤ trajectory_return M r₂ τ := by
  have h_term : ∀ t : ℕ, M.γ ^ t * r₁ (τ t) ≤ M.γ ^ t * r₂ (τ t) := fun t =>
    mul_le_mul_of_nonneg_left (h_le t) (pow_nonneg (le_of_lt M.hγ_pos) t)
  exact Summable.tsum_le_tsum h_term h_sum₁ h_sum₂

/-- Discounted trajectory returns are summable whenever rewards along the trajectory are bounded. -/
lemma summable_trajectory_return_of_bounded (M : MDP) (r : M.S → ℝ) (τ : Trace M.S) (B : ℝ)
    (h_bound : ∀ t : ℕ, |r (τ t)| ≤ B) :
    Summable (fun t : ℕ => M.γ ^ t * r (τ t)) := by
  have hγ_nonneg : 0 ≤ M.γ := le_of_lt M.hγ_pos
  have hγ_lt : M.γ < 1 := M.hγ_lt_one
  have h_geom := summable_geometric_of_lt_one hγ_nonneg hγ_lt
  have h_norm_sum : Summable (fun t : ℕ => M.γ ^ t * B) := h_geom.mul_right B
  have h_norm_le : ∀ t : ℕ, ‖M.γ ^ t * r (τ t)‖ ≤ M.γ ^ t * B := by
    intro t
    rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_nonneg hγ_nonneg]
    exact mul_le_mul_of_nonneg_left (h_bound t) (pow_nonneg hγ_nonneg t)
  exact Summable.of_norm_bounded h_norm_sum h_norm_le

/-- Upper bound for trajectory returns with rewards in a bounded interval [R_min, R_trans].
    Shows trajectory return is at most R_trans / (1 - γ). -/
theorem trapped_trajectory_return_le_of_interval (M : MDP) (r : M.S → ℝ) (τ : Trace M.S) (R_min R_trans : ℝ)
    (h_min : ∀ t : ℕ, R_min ≤ r (τ t))
    (h_max : ∀ t : ℕ, r (τ t) ≤ R_trans) :
    trajectory_return M r τ ≤ R_trans / (1 - M.γ) := by
  set B := max (|R_min|) (|R_trans|)
  have h_bound : ∀ t : ℕ, |r (τ t)| ≤ B := by
    intro t
    rw [abs_le]
    constructor
    · have h1 : -B ≤ -|R_min| := by linarith [le_max_left (|R_min|) (|R_trans|)]
      have h2 : -|R_min| ≤ R_min := neg_abs_le R_min
      linarith [h_min t]
    · have h1 : R_trans ≤ |R_trans| := le_abs_self R_trans
      have h2 : |R_trans| ≤ B := le_max_right (|R_min|) (|R_trans|)
      linarith [h_max t]
  have hγ_nonneg : 0 ≤ M.γ := le_of_lt M.hγ_pos
  have hγ_lt : M.γ < 1 := M.hγ_lt_one
  have h_geom := summable_geometric_of_lt_one hγ_nonneg hγ_lt
  have hg_sum : Summable (fun t : ℕ => M.γ ^ t * R_trans) := h_geom.mul_right R_trans
  have h_norm_sum : Summable (fun t : ℕ => M.γ ^ t * B) := h_geom.mul_right B
  have h_norm_le : ∀ t : ℕ, ‖M.γ ^ t * r (τ t)‖ ≤ M.γ ^ t * B := by
    intro t
    rw [Real.norm_eq_abs, abs_mul, abs_pow, abs_of_nonneg hγ_nonneg]
    exact mul_le_mul_of_nonneg_left (h_bound t) (pow_nonneg hγ_nonneg t)
  have h_sum : Summable (fun t : ℕ => M.γ ^ t * r (τ t)) :=
    Summable.of_norm_bounded h_norm_sum h_norm_le
  have h_le_term : ∀ t : ℕ, M.γ ^ t * r (τ t) ≤ M.γ ^ t * R_trans := fun t =>
    mul_le_mul_of_nonneg_left (h_max t) (pow_nonneg hγ_nonneg t)
  have h_tsum_le := Summable.tsum_le_tsum h_le_term h_sum hg_sum
  have h_geom_sum : (∑' t : ℕ, M.γ ^ t * R_trans) = R_trans / (1 - M.γ) := by
    rw [tsum_mul_right, tsum_geometric_of_lt_one hγ_nonneg hγ_lt, mul_comm, div_eq_mul_inv]
  dsimp [trajectory_return]
  rw [← h_geom_sum]
  exact h_tsum_le

/-- Trapped trajectory return bound specialized to task set S_trans = S_safe \ S_g. -/
theorem trapped_in_trans_return_le (M : MDP) (sets : TaskSets M) (asm : PaperAssumptions M sets)
    (r : M.S → ℝ) (τ : Trace M.S)
    (hr_min : ∀ s ∈ sets.S_safe, asm.R_min ≤ r s)
    (hr_max : ∀ s ∈ sets.S_trans, r s ≤ asm.R_trans)
    (h_trapped : ∀ t : ℕ, τ t ∈ sets.S_trans) :
    trajectory_return M r τ ≤ asm.R_trans / (1 - M.γ) := by
  have h_min_t : ∀ t : ℕ, asm.R_min ≤ r (τ t) := by
    intro t
    exact hr_min (τ t) (h_trapped t).1
  have h_max_t : ∀ t : ℕ, r (τ t) ≤ asm.R_trans := by
    intro t
    exact hr_max (τ t) (h_trapped t)
  exact trapped_trajectory_return_le_of_interval M r τ asm.R_min asm.R_trans h_min_t h_max_t

/-- Backward-compatibility alias for trapped_in_trans_return_le. -/
abbrev trapped_in_T_return_le := trapped_in_trans_return_le

/-- Trapping contradicts Bellman optimality of optimal trajectories. -/
theorem trapped_contradicts_optimality (M : MDP) (r : M.S → ℝ) (τ : Trace M.S)
    (V_star : ValueFunction M) (V_path R_trans : ℝ)
    (h_opt_traj : IsOptimalTrajectory M r V_star τ)
    (h_opt_ge_path : V_star (τ 0) ≥ V_path)
    (h_path_gt_trapped : V_path > R_trans / (1 - M.γ))
    (h_trapped_bound : trajectory_return M r τ ≤ R_trans / (1 - M.γ)) : False := by
  have h_ret : trajectory_return M r τ = V_star (τ 0) := h_opt_traj
  have h_val_le : V_star (τ 0) ≤ R_trans / (1 - M.γ) := by linarith
  linarith

end LeanProofs.Foundations

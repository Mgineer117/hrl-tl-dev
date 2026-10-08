import Mathlib
import LeanProofs.Foundations.LTL
import LeanProofs.Foundations.MDP
import LeanProofs.Foundations.Reward
import LeanProofs.Foundations.Sets
import LeanProofs.Foundations.Assumptions

set_option linter.style.header false
set_option linter.style.longLine false
set_option linter.style.emptyLine false
set_option linter.unusedVariables true

/-!
# Lemma 2: Eventual Reachability ($\mathcal{F} \psi_{\mathrm{g}}$) — Paper §4.1, Lemma 2

Establishes that for a topologically feasible MDP with maximum hitting time $T_{\mathrm{max}} < \infty$,
there exists a discount factor threshold $\gamma_{\mathrm{min}} \in [0, 1)$ such that for all
$\gamma \in (\gamma_{\mathrm{min}}, 1)$, the optimal policy $\pi_\phi^*$ satisfies eventual reachability
almost surely: $\exists t \ge 0, \tau_t \in \mathcal{S}_{\mathrm{g}}$ (i.e. $\tau \models \mathcal{F} \psi_{\mathrm{g}}$).

### Key Theorems
- `gamma_min`: Definition of threshold $\gamma_{\mathrm{min}} = \left(\frac{R_{\mathrm{trans}} - R_{\min}}{(R_{\mathrm{trans}} - R_{\min}) + d}\right)^{1/T_{\max}}$.
- `gamma_min_in_unit_interval`: Proves $\gamma_{\mathrm{min}} \in [0, 1)$.
- `gamma_pow_gt_ratio`: Shows $\gamma^T \ge \gamma^{T_{\max}} > \frac{\Delta}{\Delta + d}$ for $T \le T_{\max}$.
- `return_path_dominates_return_trapped`: Algebraic return dominance $V_{\mathrm{path}} > V_{\mathrm{trans}}$.
- `optimal_trajectory_not_trapped`: Shows trapping indefinitely in $\mathcal{S}_{\mathrm{trans}} = \mathcal{S}_{\mathrm{safe}} \setminus \mathcal{S}_{\mathrm{g}}$ contradicts Bellman optimality.
- `eventual_reachability_lemma2`: Main eventual reachability theorem (Paper Lemma 2).
- `eventual_reachability_lemma2_topological`: Grounded eventual reachability via Assumption 2 topological feasibility.
- `eventual_reachability_lemma2_deterministic`: Grounded eventual reachability under deterministic transitions.
-/

namespace LeanProofs.Proofs

open LeanProofs.Foundations

variable {M : MDP} {sets : TaskSets M}

/-- Finite geometric sum formula: ∑_{i=0}^{T-1} γ^i = (1 - γ^T) / (1 - γ). -/
lemma geom_sum_range (γ : ℝ) (hγ : γ ≠ 1) (T : ℕ) :
    ∑ i ∈ Finset.range T, γ ^ i = (1 - γ ^ T) / (1 - γ) := by
  have h := geom_sum_eq hγ T
  rw [h]
  have h_num : γ ^ T - 1 = - (1 - γ ^ T) := by ring
  have h_den : γ - 1 = - (1 - γ) := by ring
  rw [h_num, h_den, neg_div_neg_eq]

/-- Tail geometric sum formula: ∑'_{i=0}^∞ γ^(i+T) = γ^T / (1 - γ). -/
lemma geom_sum_tail (γ : ℝ) (hγ_nonneg : 0 ≤ γ) (hγ_lt : γ < 1) (T : ℕ) :
    (∑' i : ℕ, γ ^ (i + T)) = γ ^ T / (1 - γ) := by
  have h_pow : (fun i : ℕ => γ ^ (i + T)) = (fun i : ℕ => γ ^ T * γ ^ i) := by
    funext i
    rw [pow_add, mul_comm]
  rw [h_pow, tsum_mul_left]
  rw [tsum_geometric_of_lt_one hγ_nonneg hγ_lt]
  ring

/-- Decomposition of discounted return into finite prefix and infinite tail. -/
lemma trajectory_return_split (M : MDP) (r : M.S → ℝ) (τ : Trace M.S) (T : ℕ)
    (h_sum : Summable (fun t : ℕ => M.γ ^ t * r (τ t))) :
    trajectory_return M r τ =
      (∑ t ∈ Finset.range T, M.γ ^ t * r (τ t)) + (∑' t : ℕ, M.γ ^ (t + T) * r (τ (t + T))) := by
  dsimp [trajectory_return]
  exact (Summable.sum_add_tsum_nat_add T h_sum).symm

/-- Lower bound on finite prefix return of a trajectory. -/
lemma prefix_return_ge (M : MDP) (r : M.S → ℝ) (τ : Trace M.S) (T : ℕ) (R_min : ℝ)
    (hr_prefix : ∀ t < T, r (τ t) ≥ R_min) :
    ∑ t ∈ Finset.range T, M.γ ^ t * r (τ t) ≥ R_min * ((1 - M.γ ^ T) / (1 - M.γ)) := by
  have h_term : ∀ t ∈ Finset.range T, M.γ ^ t * R_min ≤ M.γ ^ t * r (τ t) := by
    intro t ht
    have ht_lt : t < T := Finset.mem_range.mp ht
    have h_pow_nonneg : 0 ≤ M.γ ^ t := pow_nonneg (le_of_lt M.hγ_pos) t
    exact mul_le_mul_of_nonneg_left (hr_prefix t ht_lt) h_pow_nonneg
  have h_sum_le := Finset.sum_le_sum h_term
  have h_mul : (∑ t ∈ Finset.range T, M.γ ^ t * R_min) = R_min * ∑ t ∈ Finset.range T, M.γ ^ t := by
    rw [← Finset.sum_mul]
    ring
  rw [h_mul] at h_sum_le
  have h_ne : M.γ ≠ 1 := ne_of_lt M.hγ_lt_one
  rw [geom_sum_range M.γ h_ne T] at h_sum_le
  exact h_sum_le

/-- Lower bound on infinite tail return of a reaching trajectory. -/
lemma tail_return_ge (M : MDP) (r : M.S → ℝ) (τ : Trace M.S) (T : ℕ) (R_goal : ℝ)
    (hr_tail : ∀ t ≥ T, r (τ t) ≥ R_goal)
    (h_sum_tail : Summable (fun t : ℕ => M.γ ^ (t + T) * r (τ (t + T)))) :
    (∑' t : ℕ, M.γ ^ (t + T) * r (τ (t + T))) ≥ R_goal * (M.γ ^ T / (1 - M.γ)) := by
  have h_geom := summable_geometric_of_lt_one (le_of_lt M.hγ_pos) M.hγ_lt_one
  have h_pow_comm : (fun t : ℕ => M.γ ^ (t + T) * R_goal) = (fun t : ℕ => (R_goal * M.γ ^ T) * M.γ ^ t) := by
    funext t
    rw [pow_add]
    ring
  have h_sum_lb : Summable (fun t : ℕ => M.γ ^ (t + T) * R_goal) := by
    rw [h_pow_comm]
    exact h_geom.mul_left (R_goal * M.γ ^ T)
  have h_term : ∀ t : ℕ, M.γ ^ (t + T) * R_goal ≤ M.γ ^ (t + T) * r (τ (t + T)) := by
    intro t
    have h_ge : t + T ≥ T := Nat.le_add_left T t
    have h_pow_nonneg : 0 ≤ M.γ ^ (t + T) := pow_nonneg (le_of_lt M.hγ_pos) _
    exact mul_le_mul_of_nonneg_left (hr_tail (t + T) h_ge) h_pow_nonneg
  have h_tsum_le := Summable.tsum_le_tsum h_term h_sum_lb h_sum_tail
  have h_eval : (∑' t : ℕ, M.γ ^ (t + T) * R_goal) = R_goal * (M.γ ^ T / (1 - M.γ)) := by
    rw [h_pow_comm, tsum_mul_left]
    rw [tsum_geometric_of_lt_one (le_of_lt M.hγ_pos) M.hγ_lt_one]
    ring
  rw [h_eval] at h_tsum_le
  exact h_tsum_le

/-- Full geometric lower bound on the trajectory return of a reaching path.
    If rewards satisfy r(τ t) ≥ R_min for t < T and r(τ t) ≥ R_goal for t ≥ T,
    then the discounted return satisfies:
    trajectory_return M r τ ≥ R_min * ((1 - γ^T) / (1 - γ)) + R_goal * (γ^T / (1 - γ)). -/
theorem path_return_lower_bound (M : MDP) (r : M.S → ℝ) (τ : Trace M.S) (T : ℕ) (R_min R_goal : ℝ)
    (hr_prefix : ∀ t < T, r (τ t) ≥ R_min)
    (hr_tail : ∀ t ≥ T, r (τ t) ≥ R_goal)
    (h_sum : Summable (fun t : ℕ => M.γ ^ t * r (τ t))) :
    trajectory_return M r τ ≥
      R_min * ((1 - M.γ ^ T) / (1 - M.γ)) + R_goal * (M.γ ^ T / (1 - M.γ)) := by
  have h_split := trajectory_return_split M r τ T h_sum
  have h_prefix := prefix_return_ge M r τ T R_min hr_prefix
  have h_tail_sum : Summable (fun t : ℕ => M.γ ^ (t + T) * r (τ (t + T))) :=
    (summable_nat_add_iff T).mpr h_sum
  have h_tail := tail_return_ge M r τ T R_goal hr_tail h_tail_sum
  linarith

/-- Grounding of optimal value bound over an admissible reaching path via Bellman optimality. -/
theorem opt_val_ge_path_of_reaching_traj
    (asm : PaperAssumptions M sets)
    (s₀ : M.S) (τ_path : Trace M.S) (T : ℕ)
    (r : M.S → ℝ)
    (hr_min : ∀ s ∈ sets.S_safe, asm.R_min ≤ r s)
    (hr_goal : ∀ s ∈ sets.S_g, asm.R_trans + asm.d ≤ r s)
    (h_safe : ∀ t < T, τ_path t ∈ sets.S_safe)
    (h_reach : ∀ t ≥ T, τ_path t ∈ sets.S_g)
    (h_sum : Summable (fun t : ℕ => M.γ ^ t * r (τ_path t)))
    (V_star : ValueFunction M)
    (h_bellman_dom : V_star s₀ ≥ trajectory_return M r τ_path) :
    V_star s₀ ≥ asm.R_min * ((1 - M.γ ^ T) / (1 - M.γ)) + (asm.R_trans + asm.d) * (M.γ ^ T / (1 - M.γ)) := by
  have hr_prefix : ∀ t < T, r (τ_path t) ≥ asm.R_min := fun t ht => hr_min (τ_path t) (h_safe t ht)
  have hr_tail : ∀ t ≥ T, r (τ_path t) ≥ asm.R_trans + asm.d := fun t ht => hr_goal (τ_path t) (h_reach t ht)
  have h_path_lb := path_return_lower_bound M r τ_path T asm.R_min (asm.R_trans + asm.d) hr_prefix hr_tail h_sum
  linarith

/-- The discount factor threshold γ_min = ( (R_trans - R_min) / ((R_trans - R_min) + d) )^(1 / T_max).
    Paper Eq. (18) / L516-L518. -/
noncomputable def gamma_min (asm : PaperAssumptions M sets) : ℝ :=
  let Δ := asm.R_trans - asm.R_min
  (Δ / (Δ + asm.d)) ^ ((1 : ℝ) / (asm.T_max : ℝ))

/-- Lemma 2 (Threshold properties): γ_min ∈ [0, 1) (Paper L517). -/
theorem gamma_min_in_unit_interval (asm : PaperAssumptions M sets) :
    0 ≤ gamma_min asm ∧ gamma_min asm < 1 := by
  dsimp [gamma_min]
  set Δ := asm.R_trans - asm.R_min
  have hΔ_nonneg : 0 ≤ Δ := by linarith [asm.hR_min_le_R_trans]
  have hd_pos : 0 < asm.d := asm.hd_pos
  have hdenom_pos : 0 < Δ + asm.d := by linarith
  have hratio_nonneg : 0 ≤ Δ / (Δ + asm.d) := div_nonneg hΔ_nonneg (le_of_lt hdenom_pos)
  have hratio_lt_one : Δ / (Δ + asm.d) < 1 := by
    rw [div_lt_one hdenom_pos]
    linarith
  have hT_pos : (0 : ℝ) < (asm.T_max : ℝ) := Nat.cast_pos.mpr asm.hT_max_pos
  have h_exp_pos : 0 < (1 : ℝ) / (asm.T_max : ℝ) := div_pos one_pos hT_pos
  constructor
  · exact Real.rpow_nonneg hratio_nonneg _
  · exact Real.rpow_lt_one hratio_nonneg hratio_lt_one h_exp_pos

/-- Lemma 2 (Discount Factor Power Dominance, Paper L544):
    Because γ ∈ (γ_min, 1) and T ≤ T_max,
    γ^T ≥ γ^(T_max) > (R_trans - R_min) / ((R_trans - R_min) + d). -/
theorem gamma_pow_gt_ratio (asm : PaperAssumptions M sets)
    (γ : ℝ) (hγ_pos : 0 < γ) (hγ_gt : γ > gamma_min asm) (hγ_lt : γ < 1)
    (T : ℕ) (hT_le : T ≤ asm.T_max) :
    γ ^ T ≥ γ ^ asm.T_max ∧
    γ ^ (asm.T_max : ℝ) > (asm.R_trans - asm.R_min) / ((asm.R_trans - asm.R_min) + asm.d) := by
  set Δ := asm.R_trans - asm.R_min
  have hΔ_nonneg : 0 ≤ Δ := by linarith [asm.hR_min_le_R_trans]
  have hdenom_pos : 0 < Δ + asm.d := by linarith [asm.hd_pos]
  have hratio_nonneg : 0 ≤ Δ / (Δ + asm.d) := div_nonneg hΔ_nonneg (le_of_lt hdenom_pos)
  have hT_pos : (0 : ℝ) < (asm.T_max : ℝ) := Nat.cast_pos.mpr asm.hT_max_pos

  -- 1. γ^T ≥ γ^(T_max) for γ ∈ (0, 1) and T ≤ T_max
  have h_decr : γ ^ asm.T_max ≤ γ ^ T :=
    pow_le_pow_of_le_one (le_of_lt hγ_pos) (le_of_lt hγ_lt) hT_le

  -- 2. γ^(T_max) > (γ_min)^(T_max) = Δ / (Δ + d)
  have h_gmin_nonneg : 0 ≤ gamma_min asm := (gamma_min_in_unit_interval asm).1
  have h_rpow_lt : (gamma_min asm) ^ (asm.T_max : ℝ) < γ ^ (asm.T_max : ℝ) :=
    Real.rpow_lt_rpow h_gmin_nonneg hγ_gt hT_pos
  -- Simplify (gamma_min asm) ^ (T_max)
  have h_gmin_eval : (gamma_min asm) ^ (asm.T_max : ℝ) = Δ / (Δ + asm.d) := by
    dsimp [gamma_min]
    rw [← Real.rpow_mul hratio_nonneg]
    have h_cancel : (1 : ℝ) / (asm.T_max : ℝ) * (asm.T_max : ℝ) = 1 :=
      one_div_mul_cancel (ne_of_gt hT_pos)
    rw [h_cancel, Real.rpow_one]
  rw [h_gmin_eval] at h_rpow_lt
  exact ⟨h_decr, h_rpow_lt⟩

/-- Lemma 2 (Algebraic Return Dominance, Paper L539-L545):
    If γ^T > (R_trans - R_min) / ((R_trans - R_min) + d) and γ ∈ (0, 1),
    then V_path(s_0) > V_trans(s_0). -/
theorem return_path_dominates_return_trapped (asm : PaperAssumptions M sets)
    (γ : ℝ) (hγ_lt : γ < 1) (γ_pow_T : ℝ)
    (h_ratio : (asm.R_trans - asm.R_min) / ((asm.R_trans - asm.R_min) + asm.d) < γ_pow_T) :
    asm.R_min * ((1 - γ_pow_T) / (1 - γ)) + (asm.R_trans + asm.d) * (γ_pow_T / (1 - γ)) >
      asm.R_trans / (1 - γ) := by
  set Δ := asm.R_trans - asm.R_min
  have hdenom_pos : 0 < Δ + asm.d := by linarith [asm.hR_min_le_R_trans, asm.hd_pos]
  have h_one_sub_γ : 0 < 1 - γ := by linarith
  have h_lin : asm.R_min * (1 - γ_pow_T) + (asm.R_trans + asm.d) * γ_pow_T > asm.R_trans := by
    have h_mul := (div_lt_iff₀ hdenom_pos).mp h_ratio
    dsimp [Δ] at h_mul
    linarith
  have h_div : (asm.R_min * (1 - γ_pow_T) + (asm.R_trans + asm.d) * γ_pow_T) / (1 - γ) > asm.R_trans / (1 - γ) :=
    div_lt_div_of_pos_right h_lin h_one_sub_γ
  have h_rearr : asm.R_min * ((1 - γ_pow_T) / (1 - γ)) + (asm.R_trans + asm.d) * (γ_pow_T / (1 - γ)) =
      (asm.R_min * (1 - γ_pow_T) + (asm.R_trans + asm.d) * γ_pow_T) / (1 - γ) := by
    ring
  rw [h_rearr]
  exact h_div

/-- Lemma 2 (Bellman Optimality Contradiction, Paper L546-L548):
    Trapping in S_trans indefinitely yields V_trans < V* ≤ V*, a contradiction.
    Therefore the optimal trajectory cannot remain trapped in S_trans. -/
theorem trapping_contradicts_bellman_optimality (V_star V_trans V_path : ℝ)
    (h_bellman_opt : V_star ≥ V_path)
    (h_path_dom : V_path > V_trans)
    (h_trapped : V_star ≤ V_trans) : False := by
  linarith

/-- Lemma 2 (Non-Trapping Derivation via Bellman Return Contradiction, Paper L524-L548):
    An optimal trajectory cannot remain trapped in S_trans indefinitely.
    Derives the Bellman return contradiction:
    V(τ) ≤ R_trans / (1 - γ) < V_path ≤ V*(s₀)
    using trapped_in_trans_return_le, return_path_dominates_return_trapped, and trapped_contradicts_optimality. -/
theorem optimal_trajectory_not_trapped
    (asm : PaperAssumptions M sets)
    (γ : ℝ) (hγ_gt : γ > gamma_min asm) (hγ_lt : γ < 1) (hγ_eq : γ = M.γ)
    (τ : Trace M.S)
    (r : M.S → ℝ)
    (hr_min : ∀ s ∈ sets.S_safe, asm.R_min ≤ r s)
    (hr_max : ∀ s ∈ sets.S_trans, r s ≤ asm.R_trans)
    (T : ℕ) (hT_le : T ≤ asm.T_max)
    (V_star : ValueFunction M)
    (h_opt_traj : IsOptimalTrajectory M r V_star τ)
    (h_opt_ge_path : V_star (τ 0) ≥
      asm.R_min * ((1 - γ ^ T) / (1 - γ)) + (asm.R_trans + asm.d) * (γ ^ T / (1 - γ))) :
    ¬ (∀ t, τ t ∈ sets.S_trans) := by
  subst hγ_eq
  have h_dom := gamma_pow_gt_ratio asm M.γ M.hγ_pos hγ_gt M.hγ_lt_one T hT_le
  have h_cast : M.γ ^ (asm.T_max : ℝ) = M.γ ^ asm.T_max := Real.rpow_natCast M.γ asm.T_max
  rw [h_cast] at h_dom
  have h_ratio : (asm.R_trans - asm.R_min) / ((asm.R_trans - asm.R_min) + asm.d) < M.γ ^ T := by
    linarith [h_dom.1, h_dom.2]
  have h_path_dom : asm.R_min * ((1 - M.γ ^ T) / (1 - M.γ)) + (asm.R_trans + asm.d) * (M.γ ^ T / (1 - M.γ)) >
      asm.R_trans / (1 - M.γ) :=
    return_path_dominates_return_trapped asm M.γ M.hγ_lt_one (M.γ ^ T) h_ratio
  intro h_trapped
  have h_trapped_bound : trajectory_return M r τ ≤ asm.R_trans / (1 - M.γ) :=
    trapped_in_trans_return_le M sets asm r τ hr_min hr_max h_trapped
  exact trapped_contradicts_optimality M r τ V_star _ asm.R_trans
    h_opt_traj h_opt_ge_path h_path_dom h_trapped_bound

/-- Lemma 2 (Eventual Reachability F ψ_g, Paper L512-L551):
    Under topological feasibility and γ > γ_min, the optimal policy
    cannot remain trapped in S_trans indefinitely and must eventually reach S_g.
    Derives non-trapping via Bellman return contradiction without assuming h_not_trapped. -/
theorem eventual_reachability_lemma2
    (asm : PaperAssumptions M sets)
    (γ : ℝ) (hγ_gt : γ > gamma_min asm) (hγ_lt : γ < 1)
    (τ : Trace M.S)
    (r : M.S → ℝ)
    (hr_min : ∀ s ∈ sets.S_safe, asm.R_min ≤ r s)
    (hr_max : ∀ s ∈ sets.S_safe, s ∉ sets.S_g → r s ≤ asm.R_trans)
    (T : ℕ) (hT_le : T ≤ asm.T_max)
    (V_star : ValueFunction M)
    (h_opt_traj : IsOptimalTrajectory M r V_star τ)
    (h_opt_ge_path : V_star (τ 0) ≥
      asm.R_min * ((1 - γ ^ T) / (1 - γ)) + (asm.R_trans + asm.d) * (γ ^ T / (1 - γ)))
    (h_safe : globally (fun s => s ∈ sets.S_safe) τ)
    (hγ_eq : γ = M.γ := by rfl) :
    eventually (fun s => s ∈ sets.S_g) τ := by
  have hr_max_trans : ∀ s ∈ sets.S_trans, r s ≤ asm.R_trans := by
    intro s hs
    exact hr_max s hs.1 hs.2
  have h_never_trapped : ¬ (∀ t, τ t ∈ sets.S_trans) :=
    optimal_trajectory_not_trapped asm γ hγ_gt hγ_lt hγ_eq τ r hr_min hr_max_trans T hT_le V_star h_opt_traj h_opt_ge_path
  by_contra h_not_reach
  have h_all_trans : ∀ t, τ t ∈ sets.S_trans := by
    intro t
    constructor
    · exact h_safe t
    · intro ht_g
      exact h_not_reach ⟨t, ht_g⟩
  exact h_never_trapped h_all_trans

/-- Lemma 2 (Eventual Reachability from Complete Topological Feasibility - Resolves Finding F-02):
    Directly obtains reaching path τ_path, hitting time T ≤ T_max, and path safety/reachability
    properties from Assumption 2 (asm.exists_reaching_path), discharging all path hypotheses. -/
theorem eventual_reachability_lemma2_topological
    (asm : PaperAssumptions M sets)
    (γ : ℝ) (hγ_gt : γ > gamma_min asm) (hγ_lt : γ < 1)
    (s₀ : M.S) (τ : Trace M.S)
    (π : M.S → M.A) (h_ind : InducesTrajectory M π s₀ τ)
    (r : M.S → ℝ)
    (hr_min : ∀ s ∈ sets.S_safe, asm.R_min ≤ r s)
    (hr_max : ∀ s ∈ sets.S_safe, s ∉ sets.S_g → r s ≤ asm.R_trans)
    (hr_goal : ∀ s ∈ sets.S_g, asm.R_trans + asm.d ≤ r s)
    (V_star : ValueFunction M)
    (h_bellman_dom : ∀ (τ_p : Trace M.S), τ_p 0 = s₀ → V_star s₀ ≥ trajectory_return M r τ_p)
    (h_bellman_step : ∀ t : ℕ, V_star (τ t) = r (τ t) + M.γ * V_star (τ (t + 1)))
    (B : ℝ) (h_bound : ∀ s, |V_star s| ≤ B)
    (h_path_sum : ∀ (τ_p : Trace M.S), Summable (fun t : ℕ => M.γ ^ t * r (τ_p t)))
    (h_safe : globally (fun s => s ∈ sets.S_safe) τ)
    (hγ_eq : γ = M.γ := by rfl) :
    eventually (fun s => s ∈ sets.S_g) τ := by
  have h_s0 : s₀ ∈ sets.S_safe := by
    rw [← h_ind.1]
    exact h_safe 0
  rcases asm.exists_reaching_path s₀ h_s0 with ⟨τ_path, T, hT_le, h_path_start, h_path_safe, h_path_reach⟩
  have h_dom := h_bellman_dom τ_path h_path_start
  have h_sum := h_path_sum τ_path
  have h_opt_traj : IsOptimalTrajectory M r V_star τ :=
    optimal_trace_is_optimal_trajectory M r V_star τ h_bellman_step B h_bound
  have h_s0_eq : τ 0 = s₀ := h_ind.1
  have h_opt_ge_path : V_star (τ 0) ≥
      asm.R_min * ((1 - γ ^ T) / (1 - γ)) + (asm.R_trans + asm.d) * (γ ^ T / (1 - γ)) := by
    rw [h_s0_eq, hγ_eq]
    exact opt_val_ge_path_of_reaching_traj asm s₀ τ_path T r hr_min hr_goal h_path_safe h_path_reach h_sum V_star h_dom
  exact eventual_reachability_lemma2 asm γ hγ_gt hγ_lt τ r hr_min hr_max T hT_le V_star h_opt_traj h_opt_ge_path h_safe hγ_eq

/-- Lemma 2 for Deterministic Transitions (Resolves Finding F-01):
    Under deterministic dynamics (IsDeterministicTransition M f) and optimal Bellman evaluation,
    the pointwise Bellman transversality step V*(τ t) = r(τ t) + γ V*(τ (t+1)) is derived
    constructively without requiring an ungrounded sample-path hypothesis. -/
theorem eventual_reachability_lemma2_deterministic
    (asm : PaperAssumptions M sets)
    (γ : ℝ) (hγ_gt : γ > gamma_min asm) (hγ_lt : γ < 1)
    (s₀ : M.S) (τ : Trace M.S)
    (π : M.S → M.A)
    (f : M.S → M.A → M.S)
    (h_det : IsDeterministicTransition M f)
    (h_supp : ∀ s a, StepSupported M s a (f s a))
    (h_τ_start : τ 0 = s₀)
    (h_τ_step : ∀ t, τ (t + 1) = f (τ t) (π (τ t)))
    (r : M.S → ℝ)
    (hr_min : ∀ s ∈ sets.S_safe, asm.R_min ≤ r s)
    (hr_max : ∀ s ∈ sets.S_safe, s ∉ sets.S_g → r s ≤ asm.R_trans)
    (hr_goal : ∀ s ∈ sets.S_g, asm.R_trans + asm.d ≤ r s)
    (V_star : ValueFunction M)
    (h_opt_eval : ∀ s, V_star s = r s + M.γ * M.E_next s (π s) V_star)
    (h_bellman_dom : ∀ (τ_p : Trace M.S), τ_p 0 = s₀ → V_star s₀ ≥ trajectory_return M r τ_p)
    (B : ℝ) (h_bound : ∀ s, |V_star s| ≤ B)
    (h_path_sum : ∀ (τ_p : Trace M.S), Summable (fun t : ℕ => M.γ ^ t * r (τ_p t)))
    (h_safe : globally (fun s => s ∈ sets.S_safe) τ)
    (hγ_eq : γ = M.γ := by rfl) :
    eventually (fun s => s ∈ sets.S_g) τ := by
  have h_ind : InducesTrajectory M π s₀ τ := by
    refine ⟨h_τ_start, fun t => ?_⟩
    rw [h_τ_step t]
    exact h_supp (τ t) (π (τ t))
  have h_bellman_step : ∀ t : ℕ, V_star (τ t) = r (τ t) + M.γ * V_star (τ (t + 1)) :=
    deterministic_bellman_step M f h_det r V_star π h_opt_eval τ h_τ_step
  exact eventual_reachability_lemma2_topological asm γ hγ_gt hγ_lt s₀ τ π h_ind r hr_min hr_max hr_goal
    V_star h_bellman_dom h_bellman_step B h_bound h_path_sum h_safe hγ_eq

end LeanProofs.Proofs


import Mathlib

set_option linter.style.header false
set_option linter.style.longLine false

namespace LeanProofs.Foundations

open MeasureTheory

/-- A Markov Decision Process with compact state and action spaces. -/
structure MDP where
  S : Type*
  A : Type*
  [topS : TopologicalSpace S]
  [compS : CompactSpace S]
  [topA : TopologicalSpace A]
  [compA : CompactSpace A]
  [measA : MeasurableSpace A]
  /-- Action space reference measure (e.g. counting measure for discrete actions) -/
  μA : Measure A
  /-- Discount factor γ ∈ (0, 1) -/
  γ : ℝ
  hγ_pos : 0 < γ
  hγ_lt_one : γ < 1
  /-- Expectation of a next-state value function given current state and action -/
  E_next : S → A → (S → ℝ) → ℝ
  /-- Monotonicity of expectation -/
  E_next_mono : ∀ s a (f g : S → ℝ), (∀ s', f s' ≤ g s') → E_next s a f ≤ E_next s a g
  /-- Expectation of a constant is the constant -/
  E_next_const : ∀ s a (c : ℝ), E_next s a (fun _ => c) = c
  /-- Linearity of expectation: addition -/
  E_next_add : ∀ s a (f g : S → ℝ), E_next s a (fun s' => f s' + g s') = E_next s a f + E_next s a g
  /-- Linearity of expectation: scalar multiplication -/
  E_next_smul : ∀ s a (c : ℝ) (f : S → ℝ), E_next s a (fun s' => c * f s') = c * E_next s a f
  /-- Transition reachability / support relation: whether successor state s' is realizable from (s, a) -/
  reachable : S → A → S → Prop := fun _ _ _ => True

attribute [instance] MDP.topS MDP.compS MDP.topA MDP.compA MDP.measA

/-- A deterministic or stochastic policy maps states to action distributions (or actions). -/
def Policy (M : MDP) := M.S → M.A → ℝ

/-- State-action value function (Q-function). -/
def QFunction (M : MDP) := M.S → M.A → ℝ

/-- State value function. -/
def ValueFunction (M : MDP) := M.S → ℝ

/-- An infinite trajectory / trace in state space `S` indexed by discrete time `t : ℕ`. -/
def Trace (S : Type*) := ℕ → S

/-- Standard Bellman Q-value calculation: Q(s,a) = r(s,a) + γ * E_{s'}[V(s')] -/
def bellman_q (M : MDP) (r : M.S → M.A → ℝ) (V : ValueFunction M) : QFunction M :=
  fun s a => r s a + M.γ * M.E_next s a V

/-- Transition reachability / support relation: whether state s' is realizable from s via action a. -/
def StepSupported (M : MDP) (s : M.S) (a : M.A) (s' : M.S) : Prop :=
  M.reachable s a s'

/-- A non-trivial transition support specification for an MDP:
    requires that transitions are explicitly restricted rather than trivially universal (Resolves Finding D3). -/
def MDPNonTrivialSupport (M : MDP) : Prop :=
  ∃ (s : M.S) (a : M.A) (s' : M.S), ¬ StepSupported M s a s'

/-- Consistency between reachability and an explicit successor support map (Resolves Finding D3). -/
def TransitionSupportedBy (M : MDP) (supp : M.S → M.A → Set M.S) : Prop :=
  ∀ s a s', StepSupported M s a s' ↔ s' ∈ supp s a

/-- Partition of next-state expectation across an obstacle subset S_o with probability p_fail.
    For an unsafe action a_u entering S_o with transition probability p_fail ∈ (0, 1]:
    E_{s'}[V(s')] ≤ p_fail * (sup_{S_o} V) + (1 - p_fail) * (sup_{S_safe} V). -/
def PartitionedExpectation (M : MDP) (S_o : Set M.S) (p_fail : ℝ) (s : M.S) (a : M.A) : Prop :=
  ∀ (V : ValueFunction M) (V_o_sup V_safe_sup : ℝ),
    (∀ s' ∈ S_o, V s' ≤ V_o_sup) →
    (∀ s' ∉ S_o, V s' ≤ V_safe_sup) →
    M.E_next s a V ≤ p_fail * V_o_sup + (1 - p_fail) * V_safe_sup

/-- Derivation of the Bellman Q-value upper bound for an unsafe action entering an obstacle with probability p_fail. -/
lemma bellman_q_unsafe_upper_bound (M : MDP) (r : M.S → M.A → ℝ) (V : ValueFunction M)
    (S_o : Set M.S) (p_fail : ℝ) (s : M.S) (a : M.A)
    (h_part : PartitionedExpectation M S_o p_fail s a)
    (V_o_sup V_safe_sup : ℝ)
    (h_V_o : ∀ s' ∈ S_o, V s' ≤ V_o_sup)
    (h_V_safe : ∀ s' ∉ S_o, V s' ≤ V_safe_sup) :
    bellman_q M r V s a ≤ r s a + M.γ * (p_fail * V_o_sup + (1 - p_fail) * V_safe_sup) := by
  dsimp [bellman_q]
  have h_exp := h_part V V_o_sup V_safe_sup h_V_o h_V_safe
  have h_gamma : 0 ≤ M.γ := M.hγ_pos.le
  nlinarith

/-- Derivation of the Bellman Q-value lower bound for a safe action maintaining minimum safe value. -/
lemma bellman_q_safe_lower_bound (M : MDP) (r : M.S → M.A → ℝ) (V : ValueFunction M)
    (s : M.S) (a_s : M.A) (V_safe_min : ℝ)
    (h_exp_ge : M.E_next s a_s V ≥ V_safe_min) :
    bellman_q M r V s a_s ≥ r s a_s + M.γ * V_safe_min := by
  dsimp [bellman_q]
  have h_gamma : 0 ≤ M.γ := M.hγ_pos.le
  nlinarith

/-- A two-component expectation decomposition for a transition (s, a) partitioned by S_o (Resolves Finding U3). -/
structure TwoComponentExpectation (M : MDP) (S_o : Set M.S) (s : M.S) (a : M.A) where
  p_actual : ℝ
  hp_actual_nonneg : 0 ≤ p_actual
  hp_actual_le_one : p_actual ≤ 1
  E_o : (M.S → ℝ) → ℝ
  E_safe : (M.S → ℝ) → ℝ
  h_decomp : ∀ V : ValueFunction M, M.E_next s a V = p_actual * E_o V + (1 - p_actual) * E_safe V
  h_E_o_le : ∀ (V : ValueFunction M) (c : ℝ), (∀ s' ∈ S_o, V s' ≤ c) → E_o V ≤ c
  h_E_safe_le : ∀ (V : ValueFunction M) (c : ℝ), (∀ s' ∉ S_o, V s' ≤ c) → E_safe V ≤ c

/-- Bridge Lemma (Resolves Finding U3):
    If the transition expectation decomposes into obstacle and safe components with actual
    obstacle probability p_actual ≥ p_fail, and the obstacle value upper bound does not exceed
    the safe value bound (V_o_sup ≤ V_safe_sup), then the PartitionedExpectation inequality holds. -/
theorem partitioned_expectation_of_components (M : MDP) (S_o : Set M.S) (p_fail : ℝ)
    (s : M.S) (a : M.A) (comp : TwoComponentExpectation M S_o s a)
    (hp_fail_le : p_fail ≤ comp.p_actual)
    (V : ValueFunction M) (V_o_sup V_safe_sup : ℝ)
    (h_V_o : ∀ s' ∈ S_o, V s' ≤ V_o_sup)
    (h_V_safe : ∀ s' ∉ S_o, V s' ≤ V_safe_sup)
    (h_order : V_o_sup ≤ V_safe_sup) :
    M.E_next s a V ≤ p_fail * V_o_sup + (1 - p_fail) * V_safe_sup := by
  rw [comp.h_decomp V]
  have h_o := comp.h_E_o_le V V_o_sup h_V_o
  have h_s := comp.h_E_safe_le V V_safe_sup h_V_safe
  have hp_act_nonneg := comp.hp_actual_nonneg
  have hp_safe_nonneg : 0 ≤ 1 - comp.p_actual := by linarith [comp.hp_actual_le_one]
  have h1 : comp.p_actual * comp.E_o V ≤ comp.p_actual * V_o_sup :=
    mul_le_mul_of_nonneg_left h_o hp_act_nonneg
  have h2 : (1 - comp.p_actual) * comp.E_safe V ≤ (1 - comp.p_actual) * V_safe_sup :=
    mul_le_mul_of_nonneg_left h_s hp_safe_nonneg
  have h_step : comp.p_actual * comp.E_o V + (1 - comp.p_actual) * comp.E_safe V ≤
                comp.p_actual * V_o_sup + (1 - comp.p_actual) * V_safe_sup := by linarith
  have h_diff_p : 0 ≤ comp.p_actual - p_fail := by linarith
  have h_diff_v : V_o_sup - V_safe_sup ≤ 0 := by linarith
  have h_slack : (comp.p_actual - p_fail) * (V_o_sup - V_safe_sup) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos h_diff_p h_diff_v
  have h_rearr : comp.p_actual * V_o_sup + (1 - comp.p_actual) * V_safe_sup ≤
                 p_fail * V_o_sup + (1 - p_fail) * V_safe_sup := by
    linarith [h_slack]
  exact le_trans h_step h_rearr

/-- Action expectation operator under policy π at state s:
    represents expectation E_{a ~ π(·|s)} over actions with monotonicity and linearity (Resolves Finding D4). -/
structure ActionExpectation (M : MDP) where
  E_act : M.S → (M.A → ℝ) → ℝ
  E_act_mono : ∀ s (f g : M.A → ℝ), (∀ a, f a ≤ g a) → E_act s f ≤ E_act s g
  E_act_const : ∀ s (c : ℝ), E_act s (fun _ => c) = c
  E_act_add : ∀ s (f g : M.A → ℝ), E_act s (fun a => f a + g a) = E_act s f + E_act s g
  E_act_smul : ∀ s (c : ℝ) (f : M.A → ℝ), E_act s (fun a => c * f a) = c * E_act s f

/-- Action expectation linearity for subtraction. -/
lemma E_act_sub {M : MDP} (E : ActionExpectation M) (s : M.S) (f g : M.A → ℝ) :
    E.E_act s (fun a => f a - g a) = E.E_act s f - E.E_act s g := by
  have h_add : (fun a => f a - g a) = (fun a => f a + (-1) * g a) := by
    ext a; ring
  rw [h_add, E.E_act_add, E.E_act_smul]
  ring

/-- Action expectation extraction of a constant shift. -/
lemma E_act_sub_const {M : MDP} (E : ActionExpectation M) (s : M.S) (c : ℝ) (g : M.A → ℝ) :
    E.E_act s (fun a => c - g a) = c - E.E_act s g := by
  rw [E_act_sub, E.E_act_const]

/-- Bridge Constructor (Resolves Finding D4):
    Any discrete probability distribution over actions induces a canonical ActionExpectation. -/
def discrete_action_expectation (M : MDP) [Fintype M.A] (π : M.S → M.A → ℝ)
    (h_nonneg : ∀ s a, 0 ≤ π s a)
    (h_sum : ∀ s, ∑ a : M.A, π s a = 1) : ActionExpectation M where
  E_act := fun s f => ∑ a : M.A, π s a * f a
  E_act_mono := by
    intro s f g hfg
    apply Finset.sum_le_sum
    intro a _
    exact mul_le_mul_of_nonneg_left (hfg a) (h_nonneg s a)
  E_act_const := by
    intro s c
    rw [← Finset.sum_mul, h_sum s, one_mul]
  E_act_add := by
    intro s f g
    have h_split : (fun a => π s a * (f a + g a)) = (fun a => π s a * f a + π s a * g a) := by
      ext a; ring
    rw [h_split, Finset.sum_add_distrib]
  E_act_smul := by
    intro s c f
    have h_mul : (fun a => π s a * (c * f a)) = (fun a => c * (π s a * f a)) := by
      ext a; ring
    rw [h_mul, ← Finset.mul_sum]

/-- A deterministic transition step where the next-state expectation evaluates to f(s, a). -/
def IsDeterministicTransition (M : MDP) (f : M.S → M.A → M.S) : Prop :=
  ∀ s a (V : ValueFunction M), M.E_next s a V = V (f s a)

/-- Derivation of the Bellman step equation for a deterministic optimal execution trace:
    V*(τ t) = r(τ t) + γ V*(τ (t+1)) holds directly from Bellman optimality and deterministic dynamics. -/
theorem deterministic_bellman_step (M : MDP) (f : M.S → M.A → M.S)
    (h_det : IsDeterministicTransition M f)
    (r : M.S → ℝ) (V_star : ValueFunction M) (π : M.S → M.A)
    (h_opt_eval : ∀ s, V_star s = r s + M.γ * M.E_next s (π s) V_star)
    (τ : Trace M.S) (h_step : ∀ t, τ (t + 1) = f (τ t) (π (τ t))) :
    ∀ t : ℕ, V_star (τ t) = r (τ t) + M.γ * V_star (τ (t + 1)) := by
  intro t
  rw [h_opt_eval (τ t), h_det (τ t) (π (τ t)) V_star, ← h_step t]

end LeanProofs.Foundations


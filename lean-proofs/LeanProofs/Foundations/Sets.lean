import Mathlib
import LeanProofs.Foundations.MDP

set_option linter.style.header false
set_option linter.style.longLine false

namespace LeanProofs.Foundations

variable (M : MDP)

/-- Task environment sets defined by robustness predicates. -/
structure TaskSets where
  /-- Goal predicate / set S_g ⊆ S -/
  S_g : Set M.S
  /-- Safe predicate / set S_safe ⊆ S -/
  S_safe : Set M.S

/-- Obstacle set S_o = S \ S_safe = S_safeᶜ -/
def TaskSets.S_o (sets : TaskSets M) : Set M.S := sets.S_safeᶜ

/-- Transition non-goal safe set: S_trans = S_safe \ S_g -/
def TaskSets.S_trans (sets : TaskSets M) : Set M.S := sets.S_safe \ sets.S_g

lemma obstacle_eq_safe_compl (sets : TaskSets M) :
    sets.S_o = sets.S_safeᶜ := rfl

lemma transient_eq (sets : TaskSets M) :
    sets.S_trans = sets.S_safe \ sets.S_g := rfl

/-- A state is in the safe set iff it is not in the obstacle set. -/
lemma mem_safe_iff_not_obstacle (sets : TaskSets M) (s : M.S) :
    s ∈ sets.S_safe ↔ s ∉ sets.S_o := by
  dsimp [TaskSets.S_o]
  simp

variable {M : MDP}

/-- A safe action from state s ensures all supported next states remain in S_safe. -/
def TaskSets.IsSafeAction (sets : TaskSets M) (s : M.S) (a : M.A) : Prop :=
  ∀ s', StepSupported M s a s' → s' ∈ sets.S_safe

/-- An unsafe action from state s has at least one supported next state in S_o. -/
def TaskSets.IsUnsafeAction (sets : TaskSets M) (s : M.S) (a : M.A) : Prop :=
  ∃ s', StepSupported M s a s' ∧ s' ∈ sets.S_o

lemma safe_action_of_not_unsafe (sets : TaskSets M) (s : M.S) (a : M.A)
    (h_not_unsafe : ¬ sets.IsUnsafeAction s a) :
    sets.IsSafeAction s a := by
  intro s' hs'
  by_contra h_not_safe
  have hs'_o : s' ∈ sets.S_o := by
    rw [mem_safe_iff_not_obstacle] at h_not_safe
    push Not at h_not_safe
    exact h_not_safe
  exact h_not_unsafe ⟨s', hs', hs'_o⟩

end LeanProofs.Foundations

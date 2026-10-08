from __future__ import annotations

from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)

import numpy as np
import torch
from gym_tl_tools import BaseVarValueInfoGenerator, Parser
from gymnasium.core import ObsType
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from rl_pipeline.core import ConfigReader
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    Distribution,
    MultiCategoricalDistribution,
)
from torch import Tensor

from hrl_tl.utils.io import get_class
from hrl_tl.wrappers.gc_ltl import OneHotGoalRep, Predicate

from .base import LowLevelPolicy, TLObs
from .utils import create_goal_conditioning_obs, parse_tl_spec


@runtime_checkable
class ActionSummer(Protocol):
    """Protocol for action modes or handlers that can sum actions for GC-LTL."""

    def sum_actions(
        self,
        action_1: NDArray,
        action_2: NDArray,
        weight: float = 0.5,
    ) -> NDArray:
        """Combine two actions with a given weight.

        Args:
            action_1: First action array.
            action_2: Second action array.
            weight: Weight for the first action in [0, 1].

        Returns:
            The resulting blended action array.
        """
        ...


class GCLTLCompositePolicyConfig(BaseModel):
    tl_spec: str = ""
    predicates: list[Predicate] = []
    model: PPO
    switch_threshold: float = 0.97
    action_sum_coeff: float = 0.5
    threshold_type: Literal["value", "robustness"] = "value"
    action_mode: ActionSummer | None = None
    var_value_info_generator: BaseVarValueInfoGenerator | None = None
    discrete_state_space: bool = False
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_action_mode(self) -> Self:
        if not self.discrete_state_space and self.action_mode is None:
            raise ValueError(
                "action_mode (implementing ActionSummer) must be provided when discrete_state_space is False."
            )
        return self


class GCLTLCompositePolicyConfigReader(
    BaseModel, ConfigReader[GCLTLCompositePolicyConfig]
):
    tl_spec: str = ""
    predicates: list[Predicate] = []
    model_path: str
    threshold_type: Literal["value", "robustness"] = "value"
    switch_threshold: float = 0.97
    action_sum_coeff: float = 0.8
    discrete_state_space: bool = False
    action_mode_cls: str | None = None
    action_mode_args: dict[str, Any] = {}
    var_value_info_generator_cls: str | None = None
    device: str = "cuda:0"
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_config(self) -> GCLTLCompositePolicyConfig:
        model = PPO.load(self.model_path, device=self.device)
        var_value_info_generator: BaseVarValueInfoGenerator | None = None
        if self.var_value_info_generator_cls is not None:
            gen_cls = get_class(self.var_value_info_generator_cls)
            var_value_info_generator = (
                gen_cls() if gen_cls is not None else None
            )

        action_mode: ActionSummer | None = None
        if self.action_mode_cls is not None:
            action_mode_cls = get_class(self.action_mode_cls)
            assert action_mode_cls is not None, (
                f"Cannot find action mode class {self.action_mode_cls}"
            )
            action_mode = action_mode_cls(**self.action_mode_args)
        elif not self.discrete_state_space:
            raise ValueError(
                "action_mode_cls must be specified when discrete_state_space is False."
            )

        return GCLTLCompositePolicyConfig(
            tl_spec=self.tl_spec,
            predicates=self.predicates,
            model=model,
            switch_threshold=self.switch_threshold,
            action_sum_coeff=self.action_sum_coeff,
            threshold_type=self.threshold_type,
            discrete_state_space=self.discrete_state_space,
            action_mode=action_mode,
            var_value_info_generator=var_value_info_generator,
            verbose=self.verbose,
        )


class GCLTLCompositePolicy[ObsType]:
    """
    Policy composition method from Qiu et al. (2023) "Instructing goal-conditioned reinforcement learning agents with temporal logic objectives"
    https://proceedings.neurips.cc/paper_files/paper/2023/file/7b35a69f434b5eb07ed1b1ef16ace52c-Paper-Conference.pdf
    """

    def __init__(
        self,
        tl_spec: str,
        predicates: list[Predicate],
        model: PPO,
        switch_threshold: float = 0.90,
        threshold_type: Literal["value", "robustness"] = "value",
        action_sum_coeff: float = 0.5,
        discrete_state_space: bool = False,
        action_mode: ActionSummer | None = None,
        var_value_info_generator: BaseVarValueInfoGenerator | None = None,
        parser: Parser | None = None,
        verbose: bool = False,
    ) -> None:
        self.discrete_state_space: bool = discrete_state_space
        if not self.discrete_state_space and action_mode is None:
            raise ValueError(
                "action_mode (implementing ActionSummer) must be provided when discrete_state_space is False."
            )
        self.action_mode: ActionSummer | None = action_mode
        self.atomic_predicates: list[Predicate] = predicates
        self.predicates: list[str] = sorted({p.name for p in predicates})

        self.goal_rep = OneHotGoalRep(predicates)
        self.tl_spec: str = tl_spec

        self.goals: list[str]
        self.constraints: list[str]
        self.goals, self.constraints = parse_tl_spec(tl_spec, self.predicates)

        self.parser: Parser = Parser() if parser is None else parser
        self.model: PPO = model
        self.switch_threshold: float = switch_threshold
        self.threshold_type: Literal["value", "robustness"] = threshold_type
        if self.threshold_type == "robustness":
            assert var_value_info_generator is not None
        self.var_value_info_generator: BaseVarValueInfoGenerator | None = (
            var_value_info_generator
        )
        self.verbose: bool = verbose

        self.constraint_idx_to_pred_idx: list[int] = [
            self.predicates.index(constraint) for constraint in self.constraints
        ]
        self.action_sum_coeff: float = action_sum_coeff

        # Check if all goals and constraints are in predicates

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        with torch.no_grad():
            goal_observations_np: list[np.ndarray | dict[str, np.ndarray]] = [
                create_goal_conditioning_obs(observation, goal, self.goal_rep)
                for goal in self.goals
            ]

            # Compute constraint values
            # The size of constraint_values is num_constraints x num_envs  in parallel envs
            # Otherwise, it is num_constraints x 1
            if self.constraints:
                constraint_observations_np: list[
                    np.ndarray | dict[str, np.ndarray]
                ] = [
                    create_goal_conditioning_obs(
                        observation, constraint, self.goal_rep
                    )
                    for constraint in self.constraints
                ]

                match self.threshold_type:
                    case "value":
                        # Use value function to determine constraint violation likelihood
                        # Batch all constraint observations into a single forward pass
                        constraint_observations_tensors: list[
                            Tensor | dict[str, Tensor]
                        ] = [
                            self.model.policy.obs_to_tensor(constraint_obs)[0]
                            for constraint_obs in constraint_observations_np
                        ]
                        constraint_values: NDArray[np.floating] = (
                            torch.stack(
                                [
                                    self.model.policy.predict_values(obs)
                                    for obs in constraint_observations_tensors
                                ]
                            )
                            .squeeze(-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        assert constraint_values.shape[0] == len(
                            self.constraints
                        )

                        # Find the most dangerous constraint

                        max_constraint_value: NDArray[np.floating] = np.max(
                            constraint_values, axis=0
                        )
                        most_dangerous_constraint_index: NDArray[np.int64] = (
                            np.argmax(constraint_values, axis=0)
                        )
                    case "robustness":
                        # Use LTL robustness to determine constraint violation likelihood
                        assert self.var_value_info_generator is not None
                        var_values: dict[str, float] = (
                            self.var_value_info_generator.get_var_values(
                                cast(Any, None),
                                observation,
                                {},
                            )
                        )
                        ap_rob_dict: dict[str, float] = {
                            atom_pred.name: self.parser.tl2rob(
                                atom_pred.formula, var_values
                            )
                            for atom_pred in self.atomic_predicates
                        }
                        rob_values: NDArray[np.floating] = np.array(
                            [
                                [
                                    ap_rob_dict[constraint]
                                    for constraint in self.constraints
                                ]
                            ]
                        )
                        constraint_values = rob_values
                        max_constraint_value = np.max(rob_values, axis=0)
                        most_dangerous_constraint_index = np.argmax(
                            rob_values, axis=0
                        )

                # Convert most_dangerous_constraint_index to predicate index
                most_dangerous_constraint_pred_index: NDArray[np.int64] = (
                    most_dangerous_constraint_index.copy()
                )
                most_dangerous_constraint_pred_index.put(
                    list(range(len(self.constraint_idx_to_pred_idx))),
                    self.constraint_idx_to_pred_idx,
                )

                # We assume there's only one goal — get distribution in a single
                # forward pass and derive both the action and probabilities from it.
                goal_obs_tensor: Tensor | dict[str, Tensor] = (
                    self.model.policy.obs_to_tensor(goal_observations_np[0])[0]
                )
                goal_action_distribution: Distribution = (
                    self.model.policy.get_distribution(goal_obs_tensor)
                )
                goal_action: np.ndarray = (
                    goal_action_distribution.get_actions(deterministic=True)
                    .cpu()
                    .numpy()
                    .squeeze(axis=0)
                )

                match goal_action_distribution:
                    case CategoricalDistribution():
                        goal_action_probs: Tensor = cast(
                            Tensor, goal_action_distribution.distribution.probs
                        )
                    case MultiCategoricalDistribution():
                        goal_action_prob_list: list[Tensor] = [
                            cast(Tensor, dist.probs)
                            for dist in goal_action_distribution.distribution
                        ]
                        # Each prob is in shape [num_envs, num_actions_i], we need to stack them to shape [num_envs, action_dim, num_actions_i]
                        # If a prob is shorter than others, we need to pad it with zeros
                        goal_action_probs = pad_probs(goal_action_prob_list)
                    case _:
                        raise NotImplementedError(
                            f"Unsupported distribution type: {type(goal_action_distribution)}"
                        )

                # dangerous_action ($$d$$) is the most-likely action to violate the constraint
                # Get distribution in a single forward pass — derive action, probs,
                # and safe_action all from this one distribution object.
                dangerous_constraint_obs_np: dict[str, np.ndarray] = (
                    create_goal_conditioning_obs(
                        observation,
                        most_dangerous_constraint_pred_index[0],
                        self.goal_rep,
                    )
                )
                dangerous_constraint_obs_tensor: Tensor | dict[str, Tensor] = (
                    self.model.policy.obs_to_tensor(
                        dangerous_constraint_obs_np
                    )[0]
                )
                dangerous_constraint_action_distribution: Distribution = (
                    self.model.policy.get_distribution(
                        dangerous_constraint_obs_tensor
                    )
                )
                dangerous_action: np.ndarray = (
                    dangerous_constraint_action_distribution.get_actions(
                        deterministic=True
                    )
                    .cpu()
                    .numpy()
                    .squeeze(axis=0)
                )

                dangerous_action_mask: Tensor = torch.ones_like(
                    goal_action_probs
                )
                # Set the probability of dangerous_action to 0 for each env in parallel envs
                match dangerous_action_mask.dim():
                    case 2:
                        dangerous_action_mask[:, dangerous_action.flatten()] = (
                            0.0  # type: ignore[index]
                        )
                        assert isinstance(
                            dangerous_constraint_action_distribution,
                            CategoricalDistribution,
                        )
                        dangerous_constraint_action_probs: Tensor = cast(
                            Tensor,
                            dangerous_constraint_action_distribution.distribution.probs,
                        )
                        safe_action: NDArray[np.int64] = (
                            torch.argmin(
                                dangerous_constraint_action_probs, dim=-1
                            )
                            .cpu()
                            .numpy()
                        )

                    case 3:
                        # For 3D case: [num_envs, action_dim, num_actions_i]
                        # dangerous_action has shape [num_envs, action_dim]
                        # e.g., [[9,3], [3,2]] for 2 envs and 2 action dims
                        # If dangerous_action.shape is (action_dim,), we need to expand it to (num_envs, action_dim) by repeating it for each env
                        if len(dangerous_action.shape) == 1:
                            dangerous_action = np.tile(
                                dangerous_action,
                                (dangerous_action_mask.shape[0], 1),
                            )
                        num_envs, action_dim = dangerous_action.shape

                        # Create indices for each environment and action dimension
                        env_indices = (
                            torch.arange(
                                num_envs, device=dangerous_action_mask.device
                            )
                            .unsqueeze(1)
                            .expand(-1, action_dim)
                        )
                        action_dim_indices = (
                            torch.arange(
                                action_dim, device=dangerous_action_mask.device
                            )
                            .unsqueeze(0)
                            .expand(num_envs, -1)
                        )
                        dangerous_action_tensor = torch.from_numpy(
                            dangerous_action
                        ).to(dangerous_action_mask.device)

                        # Zero out the dangerous actions for each environment
                        # dangerous_action_mask[env_i, action_dim_j, dangerous_action[env_i, action_dim_j]] = 0.0
                        dangerous_action_mask[
                            env_indices,
                            action_dim_indices,
                            dangerous_action_tensor,
                        ] = 0.0

                        # Find the safe action (argmin of the dangerous action distributions)
                        assert isinstance(
                            dangerous_constraint_action_distribution,
                            MultiCategoricalDistribution,
                        )
                        dangerous_constraint_action_prob_list: list[Tensor] = [
                            cast(Tensor, dist.probs)
                            for dist in dangerous_constraint_action_distribution.distribution
                        ]
                        # Each prob is in shape [num_envs, num_actions_i], we need to
                        # stack them to shape [num_envs, action_dim, num_actions_i]
                        dangerous_constraint_action_probs = pad_probs(
                            dangerous_constraint_action_prob_list
                        )
                        safe_action = (
                            torch.argmin(
                                dangerous_constraint_action_probs, dim=-1
                            )
                            .cpu()
                            .numpy()
                        )

                    case _:
                        raise NotImplementedError(
                            f"Unsupported dangerous_action_mask dim: {dangerous_action_mask.dim()}"
                        )

                blocked_reaching_action_tmp: NDArray[np.int64] = (
                    torch.argmax(
                        goal_action_probs * dangerous_action_mask, dim=-1
                    )
                    .cpu()
                    .numpy()
                )

                blocked_reaching_action = blocked_reaching_action_tmp

                avoidance_action: NDArray[np.int64] | NDArray[np.float64]
                match self.discrete_state_space:
                    case True:
                        avoidance_action = blocked_reaching_action
                    case False:
                        if self.action_mode is None:
                            raise ValueError(
                                "action_mode must be provided when discrete_state_space is False."
                            )
                        avoidance_action = self.action_mode.sum_actions(
                            blocked_reaching_action,
                            safe_action,
                            self.action_sum_coeff,
                        )
                        # Make sure the shape is the same as goal_action
                        avoidance_action = avoidance_action.reshape(
                            goal_action.shape
                        )

                # Action is combination of goal_action and avoidance_action, where goal_action is used if max_constraint_value < switch_threshold, otherwise avoidance_action is used
                condition = max_constraint_value < self.switch_threshold
                if condition.ndim < goal_action.ndim:
                    condition = np.expand_dims(
                        condition,
                        axis=tuple(range(condition.ndim, goal_action.ndim)),
                    )
                action: NDArray[np.int64] | NDArray[np.float64] = np.where(
                    condition,
                    goal_action,
                    avoidance_action,
                )
                # action = goal_action
                if self.verbose:
                    print(
                        f"GCRL-LTL: max_constraint_value: {max_constraint_value}, switch_threshold: {self.switch_threshold}"
                    )
                    print(f"GCRL-LTL: goal_action: {goal_action}")
                    print(f"GCRL-LTL: dangerous_action: {dangerous_action}")
                    print(f"GCRL-LTL: safe_action: {safe_action}")
                    print(
                        f"GCRL-LTL: blocked_reaching_action: {blocked_reaching_action}"
                    )
                    print(f"GCRL-LTL: avoidance_action: {avoidance_action}")
                    print(f"GCRL-LTL: selected action: {action}")
                    print(
                        f"GCRL-LTL: most_dangerous_constraint_index: {most_dangerous_constraint_index}, predicate index: {most_dangerous_constraint_pred_index}"
                    )
                    print(
                        f"GCRL-LTL: constraint_values: {constraint_values}, constraints: {self.constraints}"
                    )

            else:
                # No constraints, just use the goal-directing policy
                if self.verbose:
                    print("GCRL-LTL: No constraints, use goal-directing policy")
                action, _ = self.model.predict(
                    goal_observations_np[0], deterministic=True
                )

            return action, None


def pad_probs(probs: list[Tensor]) -> Tensor:
    max_action_dim = max(prob.shape[-1] for prob in probs)
    padded_probs = []
    for prob in probs:
        if prob.shape[-1] < max_action_dim:
            padding = torch.zeros(
                prob.shape[:-1] + (max_action_dim - prob.shape[-1],),
                device=prob.device,
            )
            padded_probs.append(torch.cat([prob, padding], dim=-1))
        else:
            padded_probs.append(prob)
    return torch.stack(padded_probs, dim=1)


class GCLTLLowLevelPolicy(
    LowLevelPolicy[
        GCLTLCompositePolicy, GCLTLCompositePolicyConfig, ObsType, NDArray
    ]
):
    policy_args_validator = GCLTLCompositePolicyConfig
    policy_args_reader = GCLTLCompositePolicyConfigReader

    def define_policy(
        self, policy_args: GCLTLCompositePolicyConfig
    ) -> GCLTLCompositePolicy:
        if isinstance(policy_args, dict):
            policy_args = self.policy_args_validator(**policy_args)
        policy = GCLTLCompositePolicy(
            tl_spec=self.tl_spec,
            predicates=self.tl_wrapper_args["atomic_predicates"],
            model=policy_args.model,
            switch_threshold=policy_args.switch_threshold,
            threshold_type=policy_args.threshold_type,
            action_sum_coeff=policy_args.action_sum_coeff,
            discrete_state_space=policy_args.discrete_state_space,
            action_mode=policy_args.action_mode,
            var_value_info_generator=policy_args.var_value_info_generator,
            verbose=policy_args.verbose,
        )
        return policy

    def act(
        self,
        obs: TLObs[ObsType],
        info: dict[str, Any] | None = None,
        current_env: Any = None,
        tl_wrapper_args: dict[str, Any] | None = None,
    ) -> NDArray:
        # Remove "aut_state" from obs if exists
        action, _ = self.policy.predict(cast(Any, obs))
        return action

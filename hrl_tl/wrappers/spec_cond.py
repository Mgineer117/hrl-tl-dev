"""Specification-conditioned multi-task environment wrapper for LTL subpolicy learning."""

from __future__ import annotations

import copy
from typing import Any, Generic, Literal

import gymnasium as gym
import numpy as np
from gym_tl_tools import (
    Automaton,
    BaseVarValueInfoGenerator,
    Predicate,
    RewardConfig,
)
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, Wrapper
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray


class SpecConditionedTLWrapper(
    Wrapper[dict[str, Any], ActType, ObsType, ActType],
    RecordConstructorArgs,
    Generic[ObsType, ActType],
):
    """Environment wrapper for multi-task LTL specification-conditioned subpolicies.

    Samples an LTL specification from a candidate set at each reset and conditions
    the agent's observation on the active specification while calculating rewards
    and termination according to the corresponding automaton.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        specifications: list[str],
        atomic_predicates: list[Predicate],
        var_value_info_generator: BaseVarValueInfoGenerator[ObsType, ActType],
        reward_config: RewardConfig | dict[str, Any] = RewardConfig(),
        step_penalty: float = 0.0,
        early_termination: bool = True,
        spec_rep: Literal["one_hot", "index"] = "one_hot",
        dict_spec_key: str = "goal_spec",
    ) -> None:
        """Initializes the specification-conditioned multi-task environment wrapper.

        Args:
            env: Base gymnasium environment.
            specifications: List of candidate LTL formula strings.
            atomic_predicates: List of Predicate instances defining atomic propositions.
            var_value_info_generator: Generator extracting numerical values from observations.
            reward_config: Reward configuration for automaton transitions and robustness.
            step_penalty: Constant penalty subtracted at each environment step.
            early_termination: Whether to terminate when reaching goal or trap states.
            spec_rep: Specification representation format ('one_hot' or 'index').
            dict_spec_key: Observation dictionary key for the specification vector.

        Raises:
            ValueError: If specifications or atomic_predicates is empty, or spec_rep is unsupported.
            TypeError: If reward_config is not a RewardConfig or dict.
        """
        RecordConstructorArgs.__init__(
            self,
            specifications=specifications,
            atomic_predicates=atomic_predicates,
            var_value_info_generator=var_value_info_generator,
            reward_config=reward_config,
            step_penalty=step_penalty,
            early_termination=early_termination,
            spec_rep=spec_rep,
            dict_spec_key=dict_spec_key,
        )
        Wrapper.__init__(self, env)

        if not specifications:
            raise ValueError("specifications list must not be empty.")
        if not atomic_predicates:
            raise ValueError("atomic_predicates list must not be empty.")
        if var_value_info_generator is None:
            raise ValueError("var_value_info_generator must not be None.")

        self.specifications: list[str] = list(specifications)
        self.atomic_predicates: list[Predicate] = list(atomic_predicates)
        self.var_value_info_generator = var_value_info_generator
        self.step_penalty: float = float(step_penalty)
        self.early_termination: bool = early_termination
        self.spec_rep: Literal["one_hot", "index"] = spec_rep
        self.dict_spec_key: str = dict_spec_key

        if isinstance(reward_config, dict):
            self.reward_config = RewardConfig(**reward_config)
        elif isinstance(reward_config, RewardConfig):
            self.reward_config = reward_config
        else:
            raise TypeError(
                f"reward_config must be RewardConfig or dict, got {type(reward_config)}"
            )

        # Pre-compile Automatons for each candidate specification.
        self.automatons: list[Automaton] = [
            Automaton(spec, self.atomic_predicates)
            for spec in self.specifications
        ]

        # Precompute specification encodings.
        num_specs = len(self.specifications)
        self.spec_encodings: list[NDArray[np.int64]] = []
        if self.spec_rep == "one_hot":
            for i in range(num_specs):
                vec = np.zeros(num_specs, dtype=np.int64)
                vec[i] = 1
                self.spec_encodings.append(vec)
            spec_space: spaces.Space = spaces.MultiDiscrete([2] * num_specs)
        elif self.spec_rep == "index":
            for i in range(num_specs):
                self.spec_encodings.append(np.array(i, dtype=np.int64))
            spec_space = spaces.Discrete(num_specs)
        else:
            raise ValueError(f"Unsupported spec_rep: '{self.spec_rep}'")

        # Build augmented observation space.
        if isinstance(env.observation_space, spaces.Dict):
            if self.dict_spec_key in env.observation_space.spaces:
                raise ValueError(
                    f"Key '{self.dict_spec_key}' already exists in observation space."
                )
            new_spaces = dict(env.observation_space.spaces)
            new_spaces[self.dict_spec_key] = spec_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Dict(
                {
                    "obs": env.observation_space,
                    self.dict_spec_key: spec_space,
                }
            )

        self.current_spec_idx: int = 0
        self.current_spec: str = self.specifications[0]
        self.current_automaton: Automaton = self.automatons[0]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resets the environment and samples or assigns the active specification.

        Args:
            seed: Optional PRNG seed.
            options: Optional dict with 'spec_idx' or 'tl_spec' to force specific specification.

        Returns:
            Tuple of (augmented_observation, info_dict).

        Raises:
            ValueError: If forced spec_idx or tl_spec is out of range or not found.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info = self._update_var_values(obs, info)

        if options is not None and "spec_idx" in options:
            spec_idx = int(options["spec_idx"])
            if spec_idx < 0 or spec_idx >= len(self.specifications):
                raise ValueError(
                    f"spec_idx {spec_idx} out of range [0, {len(self.specifications)})"
                )
            self.current_spec_idx = spec_idx
        elif options is not None and "tl_spec" in options:
            target_spec = str(options["tl_spec"])
            if target_spec not in self.specifications:
                raise ValueError(
                    f"tl_spec '{target_spec}' not in candidate specifications"
                )
            self.current_spec_idx = self.specifications.index(target_spec)
        else:
            self.current_spec_idx = int(
                self.np_random.integers(0, len(self.specifications))
            )

        self.current_spec = self.specifications[self.current_spec_idx]
        self.current_automaton = self.automatons[self.current_spec_idx]
        self.current_automaton.reset(seed=seed)

        info["spec_idx"] = self.current_spec_idx
        info["spec"] = self.current_spec
        info["success"] = False
        info["is_success"] = 0.0
        info["status"] = "intermediate"

        wrapped_obs = self._augment_observation(obs)
        return wrapped_obs, info

    def step(
        self, action: ActType
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Steps the environment and updates the active automaton state.

        Args:
            action: Action executed in the environment.

        Returns:
            Tuple of (augmented_observation, total_reward, terminated, truncated, info).
        """
        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        info = self._update_var_values(obs, info)
        info["original_reward"] = orig_reward

        aut_reward, next_aut_state = self.current_automaton.step(
            info, **self.reward_config.model_dump()
        )
        total_reward = float(aut_reward - self.step_penalty)

        if self.early_termination and next_aut_state in (
            self.current_automaton.goal_states
            + self.current_automaton.trap_states
        ):
            terminated = True

        is_goal = self.current_automaton.status == "goal"
        info["success"] = is_goal
        info["is_success"] = 1.0 if is_goal else 0.0
        info["status"] = self.current_automaton.status
        info["spec_idx"] = self.current_spec_idx
        info["spec"] = self.current_spec
        info["aut_state"] = next_aut_state

        wrapped_obs = self._augment_observation(obs)
        return wrapped_obs, total_reward, terminated, truncated, info

    def _update_var_values(
        self, obs: Any, info: dict[str, Any]
    ) -> dict[str, Any]:
        """Extracts variable values from observation and info."""
        info = copy.copy(info)
        var_values = self.var_value_info_generator.get_var_values(
            self.env, obs, info
        )
        info.update(var_values)
        return info

    def _augment_observation(self, obs: Any) -> dict[str, Any]:
        """Adds the active specification encoding to the observation dictionary."""
        encoding = self.spec_encodings[self.current_spec_idx]
        if isinstance(obs, dict):
            new_obs = copy.copy(obs)
            new_obs[self.dict_spec_key] = encoding
            return new_obs
        return {
            "obs": obs,
            self.dict_spec_key: encoding,
        }

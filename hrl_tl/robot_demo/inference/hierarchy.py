"""Trained temporal logic hierarchy orchestrating discrete robot primitives."""

from __future__ import annotations

import json
from typing import Any, TextIO

import numpy as np
from gymnasium import spaces as gym_spaces
from stable_baselines3.common import base_class

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import action, smoothing
from hrl_tl.robot_demo.world import arena, primitive
from hrl_tl.wrappers import gc_ltl, tl_meta_option
from hrl_tl.wrappers.low_level_policies import base as ll_base
from hrl_tl.wrappers.low_level_policies import utils


class RobotHierarchy:
    """The trained meta controller and native primitive state for one episode.

    Attributes:
        reason: Episode termination reason, or an empty string.
        display: Latest task and option information for the stage display.
        decision: Metadata for the last issued robot movement.
    """

    def __init__(
        self,
        layout: arena.Arena,
        upper: base_class.BaseAlgorithm,
        wrapper_kwargs: dict[str, Any],
        log: TextIO,
        max_actions: int | None,
        *,
        continuous: bool = False,
        smoothing_filter: smoothing.ActionSmoothingFilter | None = None,
    ) -> None:
        """Initializes the robot hierarchy and validates space compatibility.

        Args:
            layout: Resolved arena configuration.
            upper: Pretrained high-level PPO algorithm.
            wrapper_kwargs: Configuration dictionary for TLMetaOptionWrapper.
            log: Text file stream to log decisions and state transitions.
            max_actions: Optional upper bound on issued primitive actions.
            continuous: If True, restarts episodes indefinitely at current pose.
            smoothing_filter: Optional real-time action smoothing filter.

        Raises:
            ValueError: If max_actions is non-positive or spaces are incompatible.
        """
        if max_actions is not None and max_actions <= 0:
            raise ValueError("max_actions must be positive.")
        self._layout: arena.Arena = layout
        self._upper: base_class.BaseAlgorithm = upper
        self._wrapper_kwargs: dict[str, Any] = dict(wrapper_kwargs)
        self._continuous: bool = continuous
        self._smoothing_filter: smoothing.ActionSmoothingFilter | None = (
            smoothing_filter
        )
        self._episode_index: int = 0
        self._zone: primitive.PrimitiveZone = primitive.PrimitiveZone(
            layout, smoothing_filter=smoothing_filter
        )
        self._meta: tl_meta_option.TLMetaOptionWrapper = (
            tl_meta_option.TLMetaOptionWrapper(self._zone.env, **wrapper_kwargs)
        )
        self._meta.reset(seed=layout.seed)
        self._zone.env.action_space.seed(layout.seed)
        try:
            self._validate_spaces()
        except ValueError:
            self._zone.close()
            raise
        self._log: TextIO = log
        self._manual_limit: int | None = max_actions
        self._current_subpolicy: (
            ll_base.LowLevelPolicy[Any, Any, Any, Any] | None
        ) = None
        self._subpolicy_should_terminate: bool = False
        self._started: bool = False
        self._issued: int = 0
        self._option: dict[str, Any] = {"id": 0, "spec": None}
        self._upper_action: np.ndarray | None = None
        self._policy_args: dict[str, Any] = {}
        self.reason: str = ""
        self.decision: dict[str, Any] | None = None

    @property
    def display(self) -> dict[str, Any]:
        """Task state at the last completed primitive, plus the active option."""
        return {
            "arena_id": self._layout.identity,
            "subtask": self._zone.info.get("current_subtask"),
            "visits": self._zone.info.get("visit_counts"),
            "reason": self.reason,
            "option": self._option,
        }

    def next_action(
        self, measured: pose.Pose2D
    ) -> action.NativeMovementAction | None:
        """Completes the previous step, then advances the hierarchical policy.

        Args:
            measured: Latest corrected robot pose from motion capture.

        Returns:
            NativeMovementAction for the next step, or None if terminated.
        """
        if self.reason:
            return None

        if not self._started:
            self._zone.start(measured, evaluate=not self._continuous)
            self._started = True
        else:
            self._zone.complete(measured)
            if self._current_subpolicy is not None:
                self._meta.update_subpolicy_env(
                    self._current_subpolicy,
                    self._zone.observation,
                    self._zone.info,
                )
                if (
                    self._subpolicy_should_terminate
                    or self._current_subpolicy.is_aut_terminated
                ):
                    self.close_option()

        if self._zone.reason or self._zone.steps >= self._layout.max_steps:
            self.reason = self._zone.reason or "episode_limit"
            if self._continuous:
                self._restart_episode(measured)

        if (
            self._manual_limit is not None
            and self._issued >= self._manual_limit
            and not self._continuous
            and not self.reason
        ):
            self.reason = "episode_limit"

        if self.reason:
            self.close_option()
            return None

        selected = self._select_action()
        if selected is None and self._continuous and self.reason:
            self._restart_episode(measured)
            selected = self._select_action()
        if selected is None:
            return None

        command = self._zone.plan(np.asarray(selected))
        self._issued += 1
        self.decision = {
            "decision": self._issued,
            "episode": self._episode_index,
            "upper_action": (
                self._upper_action.tolist()
                if self._upper_action is not None
                else []
            ),
            "spec": self._option["spec"],
            "option": dict(self._option),
            "policy_args": self._policy_args,
        }
        self._log.write(
            json.dumps(
                {
                    "event": "policy",
                    **self.decision,
                    "arena_id": self._layout.identity,
                    "pose": measured.model_dump(),
                    "observation": {
                        key: np.asarray(value).tolist()
                        for key, value in self._zone.observation.items()
                    },
                    "action": command.model_dump(
                        exclude={"target", "native_target"}
                    ),
                    "target": command.target.model_dump(),
                    "native_target": command.native_target.model_dump(),
                    "target_clipped": command.target != command.native_target,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self._log.flush()
        return command

    def close_option(self) -> None:
        """Releases an active low-level policy without further inference."""
        if self._current_subpolicy is not None:
            self._current_subpolicy.delete_policy()
            self._current_subpolicy = None
        self._subpolicy_should_terminate = False

    def close(self) -> None:
        """Releases the option and native environment."""
        self.close_option()
        self._zone.close()

    def _select_action(self) -> np.ndarray | None:
        """Selects the next discrete primitive action from the hierarchy."""
        if self._current_subpolicy is None and not self._start_new_option():
            return self._zone.env.action_space.sample()

        if self._current_subpolicy is None:
            return None

        ll_action, ll_terminated, ll_truncated = self._meta.predict_subpolicy(
            self._current_subpolicy,
            self._zone.observation,
            self._zone.info,
        )
        if ll_terminated or ll_truncated:
            self._subpolicy_should_terminate = True
        return np.asarray(ll_action)

    def _start_new_option(self) -> bool:
        """Samples an option from the high-level policy and builds subpolicy."""
        self._upper_action, _ = self._upper.predict(
            self._zone.observation, deterministic=True
        )
        self._current_subpolicy, spec, self._policy_args = (
            self._meta.create_low_level_policy(
                self._upper_action,
                self._zone.observation,
                self._zone.info,
            )
        )
        self._option = {
            "id": self._option["id"] + 1,
            "spec": spec,
            "black_in_spec": "psi_b" in spec,
        }
        if self._current_subpolicy is None:
            self._subpolicy_should_terminate = True
            return False

        self._subpolicy_should_terminate = False
        return True

    def _restart_episode(self, measured: pose.Pose2D) -> None:
        """Restarts task state at the latest measured robot pose."""
        previous_reason = self.reason
        self.close_option()
        self._zone.close()
        self._episode_index += 1
        seed = self._layout.seed + self._episode_index
        self._zone = primitive.PrimitiveZone(
            self._layout, smoothing_filter=self._smoothing_filter
        )
        self._meta = tl_meta_option.TLMetaOptionWrapper(
            self._zone.env, **self._wrapper_kwargs
        )
        self._meta.reset(seed=seed)
        self._zone.env.action_space.seed(seed)
        self._zone.start(measured, evaluate=False)
        self._started = True
        self._issued = 0
        self._option = {"id": 0, "spec": None}
        self.reason = ""
        self._log.write(
            json.dumps(
                {
                    "event": "episode_restart",
                    "episode": self._episode_index,
                    "reason": previous_reason,
                    "pose": measured.model_dump(),
                },
                allow_nan=False,
            )
            + "\n"
        )
        self._log.flush()

    def _validate_spaces(self) -> None:
        """Validates that PPO and CPC observation/action spaces match arena."""
        if self._upper.observation_space != self._zone.env.observation_space:
            raise ValueError("PPO observation space does not match this arena.")
        if self._upper.action_space != self._meta.action_space:
            raise ValueError("PPO action space does not match the TL wrapper.")
        if not isinstance(
            self._zone.env.action_space, gym_spaces.MultiDiscrete
        ) or not np.array_equal(self._zone.env.action_space.nvec, [8, 5]):
            raise ValueError("Robot movement requires MultiDiscrete([8, 5]).")

        lower_args = self._meta.low_level_policy_args
        lower = lower_args["model"]
        predicates = self._meta.tl_wrapper_args["atomic_predicates"]
        representation = (
            gc_ltl.OneHotGoalRep(predicates)
            if lower_args["goal_rep"] == "one_hot"
            else gc_ltl.IndexGoalRep(predicates)
        )
        lower_obs = {
            key: value
            for key, value in self._zone.observation.items()
            if key not in self._meta.excluded_obs_keys
        }
        lower_obs["aut_state"] = np.array([0])
        goal_obs = utils.create_goal_conditioning_obs(
            lower_obs, 0, representation
        )
        if not lower.observation_space.contains(goal_obs):
            raise ValueError("CPC observation space does not match this arena.")
        if lower.action_space != self._zone.env.action_space:
            raise ValueError("CPC action space does not match this arena.")

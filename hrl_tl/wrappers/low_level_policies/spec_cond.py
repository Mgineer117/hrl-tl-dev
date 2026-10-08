"""Low-level policy wrapper executing a single learned LTL multi-task subpolicy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from gymnasium.core import ActType, Env, ObsType
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from rl_pipeline.core import ConfigReader
from sb3_soft import SDSAC
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.wrappers.low_level_policies.base import LowLevelPolicy, TLObs


class SpecCondSubpolicyConfig(BaseModel):
    """Configuration for SpecCondSubpolicy."""

    tl_spec: str = ""
    model: BaseAlgorithm
    specifications: list[str]
    dict_spec_key: str = "goal_spec"
    spec_rep: Literal["one_hot", "index"] = "one_hot"
    deterministic: bool = True
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SpecCondSubpolicyConfigReader(
    BaseModel, ConfigReader[SpecCondSubpolicyConfig]
):
    """Configuration reader loading checkpoint and candidate formulas for SpecCondSubpolicy."""

    model_path: str
    all_formulae_file_path: str = (
        "assets/formulae/fetch/cpc/all_formulae_1_cla_1_max_pred.json"
    )
    algo: str = "SDSAC"
    device: str = "cuda:0"
    dict_spec_key: str = "goal_spec"
    spec_rep: Literal["one_hot", "index"] = "one_hot"
    deterministic: bool = True
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_config(self) -> SpecCondSubpolicyConfig:
        """Loads model weights and candidate specifications, returning validated config.

        Returns:
            SpecCondSubpolicyConfig populated with the model and formula list.

        Raises:
            FileNotFoundError: If model_path or all_formulae_file_path does not exist.
            ValueError: If algo is unsupported.
            KeyError: If 'specifications' key is missing from the formulae file.
        """
        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        if self.algo == "SDSAC":
            model: BaseAlgorithm = SDSAC.load(
                str(model_file), device=self.device
            )
        elif self.algo == "PPO":
            model = PPO.load(str(model_file), device=self.device)
        else:
            raise ValueError(f"Unsupported algorithm: '{self.algo}'")

        formula_file = Path(self.all_formulae_file_path)
        if not formula_file.is_file():
            raise FileNotFoundError(
                f"Formulae file not found: {self.all_formulae_file_path}"
            )
        with formula_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "specifications" not in data:
            raise KeyError(
                f"Key 'specifications' missing in {self.all_formulae_file_path}"
            )
        specifications = list(data["specifications"])

        return SpecCondSubpolicyConfig(
            model=model,
            specifications=specifications,
            dict_spec_key=self.dict_spec_key,
            spec_rep=self.spec_rep,
            deterministic=self.deterministic,
            verbose=self.verbose,
        )


class SpecCondSubpolicy(
    LowLevelPolicy[
        BaseAlgorithm, SpecCondSubpolicyConfig, ObsType, NDArray[np.number]
    ]
):
    """Executes a pretrained specification-conditioned multi-task subpolicy."""

    policy_args_validator = SpecCondSubpolicyConfig
    policy_args_reader = SpecCondSubpolicyConfigReader

    def define_policy(
        self, policy_args: SpecCondSubpolicyConfig
    ) -> BaseAlgorithm:
        """Configures the active specification index and returns the model.

        Args:
            policy_args: Configuration containing the model and specification list.

        Returns:
            Underlying BaseAlgorithm subpolicy.

        Raises:
            ValueError: If self.tl_spec is not in the candidate specification list,
                or if spec_rep is unsupported.
        """
        if isinstance(policy_args, dict):
            policy_args = self.policy_args_validator(**policy_args)
        self.config = policy_args
        self.specifications = policy_args.specifications
        if self.tl_spec not in self.specifications:
            raise ValueError(
                f"Specification '{self.tl_spec}' not in candidate list: {self.specifications}"
            )
        self.spec_idx: int = self.specifications.index(self.tl_spec)

        num_specs = len(self.specifications)
        if policy_args.spec_rep == "one_hot":
            one_hot = np.zeros(num_specs, dtype=np.int64)
            one_hot[self.spec_idx] = 1
            self.spec_encoding: NDArray[np.int64] = one_hot
        elif policy_args.spec_rep == "index":
            self.spec_encoding = np.array(self.spec_idx, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported spec_rep: '{policy_args.spec_rep}'")

        return policy_args.model

    def act(
        self,
        obs: TLObs[ObsType],
        info: dict[str, Any] | None = None,
        current_env: Env[ObsType, ActType] | None = None,
        tl_wrapper_args: dict[str, Any] = {},
    ) -> NDArray[np.number]:
        """Predicts primitive action conditioned on the active specification.

        Args:
            obs: High-level environment observation dictionary with 'aut_state'.
            info: Optional environment info dictionary.
            current_env: Optional current environment instance.
            tl_wrapper_args: Optional temporal logic wrapper arguments.

        Returns:
            Predicted primitive action array.
        """
        if isinstance(obs, dict):
            policy_obs = copy.deepcopy(obs)
            policy_obs.pop("aut_state", None)
            policy_obs[self.config.dict_spec_key] = self.spec_encoding
        else:
            policy_obs = {
                "obs": obs,
                self.config.dict_spec_key: self.spec_encoding,
            }
        action, _ = self.policy.predict(
            policy_obs, deterministic=self.config.deterministic
        )
        return action

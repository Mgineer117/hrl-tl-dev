from __future__ import annotations

from typing import Literal, override

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized

import hrl_tl.envs.fetch  # noqa: F401
from hrl_tl.envs.fetch.env import FetchReachAvoidEnv
from hrl_tl.envs.fetch.var_value import FetchVarValueInfoGenerator


class FetchReachAvoidEnvTest(parameterized.TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.env = gym.make(
            "hrl_tl/FetchReachAvoid-v0", render_mode="rgb_array"
        )

    @override
    def tearDown(self) -> None:
        self.env.close()
        super().tearDown()

    def _get_base_env(self) -> FetchReachAvoidEnv:
        base_env = self.env.unwrapped
        assert isinstance(base_env, FetchReachAvoidEnv)
        return base_env

    def test_spaces(self) -> None:
        obs, _ = self.env.reset(seed=42)
        self.assertEqual(self.env.action_space.shape, (3,))
        self.assertIn("observation", obs)
        self.assertIn("yellow_dist", obs)

    def test_step_execution(self) -> None:
        self.env.reset(seed=42)
        action = np.array([2, 2, 2], dtype=np.int64)
        obs, reward, terminated, _, _ = self.env.step(action)
        self.assertAlmostEqual(float(reward), -0.002, places=3)
        self.assertFalse(terminated)
        self.assertNotIn("desired_goal", obs)

    def test_subtask_progression(self) -> None:
        self.env.reset(seed=42)
        base_env = self._get_base_env()
        base_env.set_agent_pos(base_env.yellow_positions[0])
        _, _, _, _, info = self.env.step(np.array([2, 2, 2], dtype=np.int64))
        self.assertEqual(info["subtask"], 1)
        self.assertEqual(base_env.current_subtask, 1)

    def test_red_obstacle_collision(self) -> None:
        self.env.reset(seed=42)
        base_env = self._get_base_env()
        base_env.set_agent_pos(base_env.red_positions[0])
        _, reward, terminated, _, _ = self.env.step(
            np.array([2, 2, 2], dtype=np.int64)
        )
        self.assertEqual(reward, -1.002)
        self.assertTrue(terminated)

    def test_goal_completion(self) -> None:
        self.env.reset(seed=42)
        base_env = self._get_base_env()
        base_env.current_subtask = 1
        base_env.set_agent_pos(base_env.white_pos)
        _, reward, terminated, _, _ = self.env.step(
            np.array([2, 2, 2], dtype=np.int64)
        )
        self.assertEqual(reward, 99.998)
        self.assertTrue(terminated)

    def test_geometry_interception(self) -> None:
        self.env.reset(seed=42)
        base_env = self._get_base_env()
        for y_pos in base_env.yellow_positions:
            mid = (y_pos + base_env.white_pos[0]) / 2.0
            min_dist_to_red = float(
                np.min(np.linalg.norm(base_env.red_positions - mid, axis=1))
            )
            self.assertLess(min_dist_to_red, 1e-4)

    def test_var_value_info_generator(self) -> None:
        obs, _ = self.env.reset(seed=42)
        generator = FetchVarValueInfoGenerator()
        info = generator.get_var_values(env=None, obs=obs)
        self.assertIn("d_y", info)
        self.assertIn("d_r", info)
        self.assertIn("d_w", info)

    @parameterized.named_parameters(
        ("default_view", "default", (480, 480, 3)),
        ("top_down_view", "top_down", (480, 480, 3)),
        ("combined_view", "combined", (480, 960, 3)),
    )
    def test_render_views(
        self,
        render_view: Literal["default", "top_down", "combined"],
        expected_shape: tuple[int, int, int],
    ) -> None:
        self.env.reset(seed=42)
        base_env = self._get_base_env()
        frame = base_env.render(render_view=render_view)
        self.assertEqual(frame.shape, expected_shape)

    def test_render_multiview(self) -> None:
        self.env.reset(seed=42)
        base_env = self._get_base_env()
        multiview = base_env.render_multiview()
        self.assertEqual(multiview["default"].shape, (480, 480, 3))
        self.assertEqual(multiview["top_down"].shape, (480, 480, 3))
        self.assertEqual(multiview["combined"].shape, (480, 960, 3))

    def test_zone_size_applied_to_site_size(self) -> None:
        self.env.reset(seed=42)
        base_env = self._get_base_env()
        base_env._render_callback()
        expected_size = base_env.reach_avoid_config.zone_config.zone_size
        sid = base_env._mujoco.mj_name2id(
            base_env.model, base_env._mujoco.mjtObj.mjOBJ_SITE, "yellow0"
        )
        np.testing.assert_allclose(
            base_env.model.site_size[sid],
            [expected_size, expected_size, expected_size],
        )

    def test_zone_displacement_sorting(self) -> None:
        obs, _ = self.env.reset(seed=42)
        for key in ("yellow_dist", "red_dist", "white_dist"):
            dists = np.linalg.norm(obs[key], axis=1)
            self.assertTrue(
                np.all(dists[:-1] <= dists[1:]),
                f"{key} not sorted in ascending order: {dists}",
            )

    def test_zone_displacement_reset_randomization(self) -> None:
        obs1, _ = self.env.reset(seed=42)
        obs2, _ = self.env.reset(seed=123)
        self.assertFalse(
            np.allclose(obs1["yellow_dist"], obs2["yellow_dist"]),
            "yellow_dist remained unchanged across resets with different seeds",
        )
        self.assertFalse(
            np.allclose(obs1["white_dist"], obs2["white_dist"]),
            "white_dist remained unchanged across resets with different seeds",
        )


if __name__ == "__main__":
    absltest.main()

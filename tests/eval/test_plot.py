from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import matplotlib.font_manager as fm
import numpy as np
from absl.testing import absltest, parameterized
from pydantic import BaseModel

from hrl_tl.eval.plot import (
    CurvePlotConfig,
    filter_configs_by_method,
    resolve_font_family,
    run_curve_pipeline,
)
from hrl_tl.eval.reward import EvalDataConfig, EvalFileReader


class _DummyReaderConfig(BaseModel):
    dummy_val: int = 1


class _DummyEvalFileReader(EvalFileReader[_DummyReaderConfig]):
    @staticmethod
    def read_eval_files(
        eval_data_config: _DummyReaderConfig, num_reps: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rewards = [np.array([10.0, 20.0, 30.0]) for _ in range(num_reps)]
        timesteps = [np.array([0, 500, 1000]) for _ in range(num_reps)]
        return rewards, timesteps


class PlotUtilsTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config_a = EvalDataConfig(
            name="MethodA",
            num_replicates=2,
            max_timesteps=1000,
            data_points=3,
            smooth_window=1,
            file_reader_class=_DummyEvalFileReader,
            eval_file_config=_DummyReaderConfig(),
        )
        self.config_b = EvalDataConfig(
            name="MethodB",
            num_replicates=2,
            max_timesteps=1000,
            data_points=3,
            smooth_window=1,
            file_reader_class=_DummyEvalFileReader,
            eval_file_config=_DummyReaderConfig(),
        )
        self.configs = (self.config_a, self.config_b)

    @parameterized.named_parameters(
        ("empty_selection", [], 2),
        ("none_selection", None, 2),
        ("single_match", ["MethodA"], 1),
        ("no_match", ["NonExistent"], 0),
    )
    def test_filter_configs_by_method(
        self, selected_methods: list[str] | None, expected_count: int
    ) -> None:
        filtered = filter_configs_by_method(self.configs, selected_methods)
        self.assertLen(filtered, expected_count)

    def test_resolve_font_family_fallback(self) -> None:
        font_entry = fm.FontEntry(name="DejaVu Serif")
        with mock.patch.object(
            fm.fontManager, "ttflist", [font_entry], spec_set=False
        ):
            font = resolve_font_family(
                preferred_font="NonExistentFont",
                candidates=("DejaVu Serif",),
            )
            self.assertEqual(font, "DejaVu Serif")

    def test_run_curve_pipeline_with_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plot_path = tmp_path / "plot.png"
            csv_path = tmp_path / "summary.csv"

            eval_df, summary_df = run_curve_pipeline(
                configs=self.configs,
                plot_path=plot_path,
                csv_path=csv_path,
                selected_methods=["MethodA"],
                plot_config=CurvePlotConfig(
                    xlim_left=0.0,
                    xlim_right=1.0,
                    dpi=100,
                ),
            )

            self.assertFalse(eval_df.empty)
            self.assertTrue(plot_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("mean_reward", summary_df.columns)
            self.assertEqual(summary_df.iloc[0]["method"], "MethodA")


if __name__ == "__main__":
    absltest.main()

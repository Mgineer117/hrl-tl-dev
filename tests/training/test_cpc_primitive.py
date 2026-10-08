from __future__ import annotations

from pathlib import Path
from unittest import mock

from absl.testing import absltest, parameterized
from rl_pipeline.sb3 import SB3ReplicatePipeline

from hrl_tl.training.cpc_primitive import run_cpc_primitive_training
from hrl_tl.training.types import ReplayModel

_TEST_CONFIG_PATH = Path(
    "configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml"
)


class CpcPrimitiveTrainingTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mock_train = self.enter_context(
            mock.patch.object(
                SB3ReplicatePipeline,
                "train_on_unsaved_model",
                autospec=True,
                spec_set=True,
                return_value=[],
            )
        )
        self.mock_load_models = self.enter_context(
            mock.patch.object(
                SB3ReplicatePipeline,
                "load_models",
                autospec=True,
                spec_set=True,
                return_value=[],
            )
        )
        self.mock_record_replays = self.enter_context(
            mock.patch.object(
                SB3ReplicatePipeline,
                "record_replays",
                autospec=True,
                spec_set=True,
            )
        )

    def test_run_cpc_primitive_training_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_cpc_primitive_training(Path("configs/non_existent.yaml"))

    @parameterized.named_parameters(
        (
            "with_replays_and_overrides",
            True,
            True,
            "final",
            "1",
            2,
            3,
        ),
        (
            "without_replays_or_overrides",
            False,
            False,
            "best",
            None,
            None,
            None,
        ),
        (
            "with_latest_replay_model",
            True,
            True,
            "latest",
            None,
            None,
            None,
        ),
    )
    def test_run_cpc_primitive_training(
        self,
        retrain_model: bool,
        record_replays: bool,
        replay_model: ReplayModel,
        device: str | None,
        num_replicates: int | None,
        replicate_start_id: int | None,
    ) -> None:
        pipeline, models = run_cpc_primitive_training(
            config_path=_TEST_CONFIG_PATH,
            retrain_model=retrain_model,
            record_replays=record_replays,
            replay_model=replay_model,
            device=device,
            num_replicates=num_replicates,
            replicate_start_id=replicate_start_id,
            verbose=False,
        )

        self.mock_train.assert_called_once_with(pipeline)
        self.mock_load_models.assert_called_once_with(pipeline, replay_model)
        if record_replays:
            self.mock_record_replays.assert_called_once_with(
                pipeline, models, verbose=False
            )
        else:
            self.mock_record_replays.assert_not_called()

        if num_replicates is not None:
            self.assertEqual(len(pipeline.ind_pipeline_configs), num_replicates)
            self.assertEqual(
                pipeline.replicate_config.num_replicates, num_replicates
            )
        if replicate_start_id is not None:
            self.assertEqual(
                pipeline.replicate_config.replicate_start_id, replicate_start_id
            )

        for ind_config in pipeline.ind_pipeline_configs:
            self.assertEqual(ind_config.retrain_model, retrain_model)
            if device is not None:
                self.assertEqual(ind_config.device, "cuda:1")


if __name__ == "__main__":
    absltest.main()

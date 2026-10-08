import datetime
import random
import uuid

import wandb
from hrl_tl.baseline.algorithms.HIRO import HIRO
from hrl_tl.baseline.algorithms.PPO import PPO
from hrl_tl.baseline.algorithms.TL_HRL import TL_HRL
from hrl_tl.baseline.utils.get_args import get_args
from hrl_tl.baseline.utils.rl import call_env
from hrl_tl.baseline.utils.utils import (
    concat_csv_columnwise_and_delete,
    override_args,
    seed_all,
    setup_logger,
)

wandb.require("core")


def run(args, seed: int, unique_id: str, exp_time: str):
    """Initiate the training process upon given args"""
    # call logger
    seed_all(seed)
    env = call_env(args)
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    algo_classes = {"PPO": PPO, "TL_HRL": TL_HRL, "HIRO": HIRO}

    alg_class = algo_classes.get(args.algo_name)
    if alg_class is None:
        raise ValueError(f"Unknown algorithm: {args.algo_name}")

    alg = alg_class(
        env=env,
        logger=logger,
        writer=writer,
        args=args,
    )
    alg.begin_training()

    wandb.finish()
    writer.close()


# === ENV LOOP === #
if __name__ == "__main__":
    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    seed_all(init_args.init_seed)
    seeds = [random.randint(1, 100_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args()
        run(args, seed, unique_id, exp_time)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)

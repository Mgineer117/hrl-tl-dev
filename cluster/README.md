# Illinois Campus Cluster (ICC) User Guide for HRL-TL

This guide describes how to run reinforcement learning training pipelines and Optuna hyperparameter tuning studies on the Illinois Campus Cluster (ICC) using Slurm.

---

## 1. Cluster Partitions & Limits

| Partition Alias | Slurm Partition | Default Account | GPU Type | Max Walltime | Concurrency Cap | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `eng` *(default)* | `eng-research-gpu` | `huytran1-ae-eng` | NVIDIA A10 | 48:00:00 (2 days) | **~21 jobs** | Engineering research GPU allocation (primary queue). |
| `ic` | `IllinoisComputes-GPU` | `huytran1-ic` | NVIDIA A100 / H200 | 72:00:00 (3 days) | **4 GPUs** | Campus Computes allocation (under `gpu4` QoS limit). |
| `secondary` | `secondary` | `huytran1-ae-eng` | Opportunistic | 04:00:00 (4 hours) | 10 | Free preemptible queue; shortest wait time, best for tests. |
| `all` | *(split)* | Both | Mixed | By partition | Combined (25) | Prioritizes `eng` (up to 21), routing overflow to `ic` (up to 4). |

> [!IMPORTANT]
> **GPU Device Ordinal Mapping**:
> When Slurm allocates 1 GPU to a job (`--gpus-per-task=1`), that GPU is always exposed to CUDA as device `0` (`cuda:0`), regardless of which physical GPU on the compute node was assigned.
> All scripts and batch files automatically pass `--device=0` so existing YAML configurations specifying `device: 1` or `device: 2` will not crash with `invalid device ordinal`.

---

## 2. Quick Start: Job Launcher CLI (`queue_experiment.sh`)

The `cluster/queue_experiment.sh` wrapper simplifies job submission and handles Slurm arguments, accounts, concurrency caps, and parallel replicate arrays.

### Training Experiments

Both the target training script and configuration YAML must be provided. By default, jobs run on `eng-research-gpu`:
```bash
# Meta-Option CPC training on Fourroom Continuous (defaults to eng-research-gpu)
./cluster/queue_experiment.sh train scripts/cpc/main_tl_hrl_meta_option_fr_cont_rep_sdsac.py \
    configs/fr_cont/train/3.j/3.j.e_hrl_train_rep3.yaml

# Train 5 replicates in parallel across GPUs
./cluster/queue_experiment.sh train scripts/cpc/train_primitive_rep.py \
    configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml --replicates=5

# Train 3 replicates starting from replicate ID 2
./cluster/queue_experiment.sh train scripts/cpc/train_primitive_zone_rep.py \
    configs/zone/cpc/pr/8.i.b_train_primitive_rep.yaml --replicates=3 --replicate-start-id=2

# GC-LTL baseline training explicitly on IllinoisComputes-GPU
./cluster/queue_experiment.sh train scripts/baseline/main_gcltl_meta_option_fr_cont_rep.py \
    configs/fr_cont/train/5.b/5.b.b_hrl_train_rep4.yaml --partition=ic

# PPO baseline training on secondary queue
./cluster/queue_experiment.sh train scripts/baseline/main_ppo_rep.py \
    configs/zone/train/6.e/6.e.b_ppo.yaml --partition=secondary
```

### Hyperparameter Tuning (Optuna)

Tuning runs execute Optuna studies and store trials in `db/<study_name>.db`. When finished, the best parameters are automatically exported to `configs/tuning/best/<study_name>_best.yaml`. The target tuning script is always required:

```bash
# Run 25 trials on fr_cont using eng-research-gpu
./cluster/queue_experiment.sh tune scripts/cpc/tune_tl_hrl_meta_option.py \
    --env=fr_cont --trials=25 --timesteps=2000000

# Run tuning on zone environment on ic partition
./cluster/queue_experiment.sh tune scripts/cpc/tune_tl_hrl_meta_option.py \
    --env=zone --trials=20 --timesteps=1500000 --partition=ic

# Run tuning and immediately train 5 replicates with the best parameters
./cluster/queue_experiment.sh tune scripts/cpc/tune_tl_hrl_meta_option.py \
    --env=fr_cont --trials=25 --train-replicates

# Alternative tuning runner (e.g. primitive tuning)
./cluster/queue_experiment.sh tune scripts/cpc/tune_primitive.py \
    --env=fr_cont --trials=20
```

### Campaigns & Manifests (Job Arrays)

To run a series of experiments in parallel across compute nodes:

1. Create a manifest file (e.g. `cluster/manifests/my_campaign.txt`) with one experiment per line:
   ```text
   scripts/cpc/main_tl_hrl_meta_option_fr_cont_rep_sdsac.py configs/fr_cont/train/3.j/3.j.e_hrl_train_rep0.yaml
   scripts/cpc/main_tl_hrl_meta_option_fr_cont_rep_sdsac.py configs/fr_cont/train/3.j/3.j.e_hrl_train_rep1.yaml
   scripts/cpc/main_tl_hrl_meta_option_fr_cont_rep_sdsac.py configs/fr_cont/train/3.j/3.j.e_hrl_train_rep2.yaml
   scripts/cpc/main_tl_hrl_meta_option_fr_cont_rep_sdsac.py configs/fr_cont/train/3.j/3.j.e_hrl_train_rep3.yaml
   scripts/cpc/main_tl_hrl_meta_option_fr_cont_rep_sdsac.py configs/fr_cont/train/3.j/3.j.e_hrl_train_rep4.yaml
   ```

2. Submit the campaign:
   ```bash
   # Submit on default partition (eng-research-gpu, concurrency capped at 21)
   ./cluster/queue_experiment.sh campaign cluster/manifests/my_campaign.txt

   # Or split across partitions: fills eng (up to 21) and routes overflow to ic (up to 4)
   ./cluster/queue_experiment.sh campaign cluster/manifests/my_campaign.txt --partition=all
   ```

### Dry-Run Mode (`--dry-run`)

Test submission arguments, partition accounts, and projected start time without submitting actual jobs:
```bash
./cluster/queue_experiment.sh train scripts/cpc/train_primitive_rep.py \
    configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml --replicates=5 --dry-run
```

### Excluding Problematic Nodes (`-x`, `--exclude`)

Exclude specific cluster nodes (e.g. degraded or faulty nodes):
```bash
./cluster/queue_experiment.sh train scripts/cpc/train_primitive_rep.py \
    configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml \
    --exclude=ccc0314,ccc0315,ccc0316
```

---

## 3. Direct `sbatch` Usage

You can also call `sbatch` directly if you prefer:

```bash
# Submit a training job (specifying --job-name routes logs to cluster/logs/train/<SCRIPT_NAME>/)
sbatch --account=huytran1-ae-eng --partition=eng-research-gpu \
    --job-name=train_primitive_rep \
    cluster/train_experiment.sbatch \
    scripts/cpc/train_primitive_rep.py \
    configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml

# Submit a tuning job (specifying --job-name routes logs to cluster/logs/tune/<SCRIPT_NAME>/)
sbatch --account=huytran1-ae-eng --partition=eng-research-gpu \
    --job-name=tune_tl_hrl_meta_option \
    cluster/tune_experiment.sbatch \
    scripts/cpc/tune_tl_hrl_meta_option.py --env=fr_cont --trials=25
```

> [!NOTE]
> When submitting directly with `sbatch`, specify `--job-name=<SCRIPT_NAME>` (e.g. script basename without `.py`) so logs are saved under `cluster/logs/{train,tune}/<SCRIPT_NAME>/`. If omitted, logs default to `cluster/logs/{train,tune}/hrl-{train,tune}/`.

---

## 4. Monitoring & Inspecting Jobs

### Check Queue Status
```bash
# Check all your running or pending jobs
squeue -u $USER

# Check detailed status of a specific job
scontrol show job <JOB_ID>
```

### View Output Logs
All job outputs are automatically organized under script-name subdirectories in `cluster/logs/`:
- Training logs: `cluster/logs/train/<SCRIPT_NAME>/<SCRIPT_NAME>_o<JOB_ID>_<ARRAY_TASK_ID>.log`
- Tuning logs: `cluster/logs/tune/<SCRIPT_NAME>/<SCRIPT_NAME>_o<JOB_ID>.log`

Follow a running job log in real time:
```bash
tail -f cluster/logs/train/<SCRIPT_NAME>/<SCRIPT_NAME>_o<JOB_ID>_*.log
# or for tuning:
tail -f cluster/logs/tune/<SCRIPT_NAME>/<SCRIPT_NAME>_o<JOB_ID>.log
```

### Cancel Jobs
```bash
# Cancel a specific job or array task
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

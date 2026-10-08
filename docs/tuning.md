# Hyperparameter Tuning & Replicate Evaluation

This document covers how to run, monitor, configure, and evaluate hyperparameter optimization (HPO) studies in HRL-TL using Optuna and `rl-pipeline`.

---

## 1. Usage & Running Tuning

All tuning runner scripts are located in `scripts/` organized by method family:
- **Proposed Method (CPC)**: `scripts/cpc/tune_primitive.py`, `scripts/cpc/tune_tl_hrl.py`
- **Baselines**:
  - GC-LTL: `scripts/baseline/tune_gcltl_primitive.py`, `scripts/baseline/tune_gcltl.py`
  - ALLO: `scripts/baseline/tune_allo_pretrain.py`, `scripts/baseline/tune_allo.py`
  - HIRO: `scripts/baseline/tune_hiro.py`
  - Flat PPO: `scripts/baseline/tune_ppo.py`

### Universal Command-Line Flags

Every tuning script accepts a common set of flags defined via `absl.flags`:

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--env` | `string` | Target environment (`fr_cont` or `zone`). | Script default |
| `--trials` | `int` | Override total trials to run (useful for quick smoke tests). | Config value (`n_trials`) |
| `--timesteps` | `int` | Override timesteps trained per trial. | Config value (`total_timesteps`) |
| `--device` | `string` | Target compute device (`1`, `cuda:1`, `cpu`). Normalized automatically. | Pipeline default |
| `--dashboard` | `bool` | Launch Optuna dashboard web server in-process during tuning. | `false` |
| `--dashboard_port` | `int` | Port for the Optuna dashboard web server. | `8080` |

### Running Tuning Scripts

Run any script by selecting the environment and optional overrides:

```bash
# 1. CPC TL-HRL (Proposed Method)
python scripts/cpc/tune_primitive.py --env fr_cont --device 1
python scripts/cpc/tune_tl_hrl.py --env fr_cont --device 1

# 2. GC-LTL Baseline
python scripts/baseline/tune_gcltl_primitive.py --env fr_cont --device 1
python scripts/baseline/tune_gcltl.py --env fr_cont --device 1

# 3. ALLO Baseline
python scripts/baseline/tune_allo_pretrain.py --env fr_cont --device 1
python scripts/baseline/tune_allo.py --env fr_cont --device 1

# 4. HIRO Baseline
python scripts/baseline/tune_hiro.py --env fr_cont --device 1

# 5. Flat PPO Baseline
python scripts/baseline/tune_ppo.py --env fr_cont --device 1
```

> [!TIP]
> **Smoke Testing**: To quickly verify that a script runs end-to-end without waiting for a full study, pass `--trials 1 --timesteps 100`:
> ```bash
> python scripts/baseline/tune_ppo.py --env fr_cont --trials 1 --timesteps 100 --device 1
> ```

---

## 2. Optuna Dashboard & Monitoring

Optuna logs trial parameters, intermediate evaluations, and metrics to an SQLite database (`db/<study_name>.db`).

### Launching the Dashboard

You can monitor studies in real time or inspect completed studies using either method:

1. **In-Process During Tuning**: Pass `--dashboard` when launching any tuning script:
   ```bash
   python scripts/baseline/tune_ppo.py --env fr_cont --dashboard --dashboard_port 8080
   ```
2. **Standalone Post-Hoc CLI**: Run `optuna-dashboard` directly pointing to any SQLite database file:
   ```bash
   optuna-dashboard sqlite:///db/ppo_fr_cont.db --host 0.0.0.0 --port 8080
   ```

Open your browser at `http://localhost:8080` (or the machine's IP address) to view the dashboard.

### Key Visualizations

The dashboard provides several diagnostic views:
- **Optimization History**: Plots objective value across trial numbers, visualizing convergence trends and trial variance.
- **Parameter Importances**: Evaluates hyperparameter influence on the objective via functional ANOVA (fANOVA) to identify dominant factors.
- **Slice & Contour Plots**: Shows one-dimensional slices and two-dimensional interactions between hyperparameters against the objective value.
- **Intermediate Values**: Displays learning curves across evaluation milestones to observe early-stopping pruning behavior.

---

## 3. How to Configure Tuning

Tuning configurations are stored under `configs/tuning/*.yaml`. Each configuration defines the Optuna study metadata, evaluation budget, and search space via `TuningWorkflowConfig`.

### Declarative YAML Schema

```yaml
study_name: ppo_fr_cont                  # Unique Optuna study name
db_filename: ppo_fr_cont.db              # SQLite database filename
db_dir: db                               # Directory to store SQLite databases
total_timesteps: 1000000                 # Training timesteps per trial (reduced horizon)
n_trials: 20                             # Total optimization trials
n_startup_trials: 5                      # Random exploration trials before TPE modeling
n_evaluations: 5                         # Intermediate evaluation checkpoints per trial
n_eval_episodes: 100                     # Evaluation episodes per checkpoint
n_jobs: 1                                # Parallel trial workers (process-level)
parallel_backend: process                # 'process' or 'thread'
launch_dashboard: false                  # Whether to auto-launch web dashboard
dashboard_port: 8080                     # Web dashboard port

direction: maximize                      # 'maximize' (default) or 'minimize'
metric: mean_reward                      # 'mean_reward' (eval) or loss key (e.g. 'loss/total')

tune_params:
  # Categorical parameter
  - name: batch_size
    target: algo_kwargs.batch_size
    suggest_type: categorical
    choices: [1000, 2000, 4000, 8000]

  # Continuous float parameter on log scale
  - name: learning_rate
    target: algo_kwargs.learning_rate
    suggest_type: float
    low: 0.00001
    high: 0.001
    log: true

  # Hierarchical wrapper parameter
  - name: max_low_level_policy_steps
    target: wrapper_kwargs.max_low_level_policy_steps
    suggest_type: categorical
    choices: [25, 40, 50, 60, 75]
```

### Parameter Target Routing

The `target` key supports dot notation that routes parameters into the underlying pipeline configuration:
- `algo_kwargs.<param>`: Injected directly into the RL algorithm kwargs (e.g., `algo_kwargs.batch_size`, `algo_kwargs.learning_rate`).
- `wrapper_kwargs.<param>`: Injected into environment wrapper configurations (e.g., `wrapper_kwargs.max_low_level_policy_steps`, `wrapper_kwargs.spec_rep_args.args.L_gain.range`).
- If no dot prefix is provided, parameters are matched against `wrapper_kwargs` first (if present), defaulting to `algo_kwargs`.

### Supported Parameter Types

| `suggest_type` | Required Fields | Optional Fields | Description |
| :--- | :--- | :--- | :--- |
| `categorical` | `choices: [...]` | — | Samples from discrete discrete choices or lists (e.g. ranges `[10.0, 60.0]`). |
| `float` | `low`, `high` | `log: bool`, `step: float` | Samples floating-point values uniformly or log-uniformly. |
| `int` | `low`, `high` | `log: bool`, `step: int` | Samples integer values. |

---

## 4. Post-Tuning: Replicate Generation & Evaluation

Once tuning concludes, the workflow automatically transitions from exploration to statistical verification:

```
Tuning Study (db/*.db)
       │
       ▼
Best Parameters Export (configs/tuning/best/*_best.yaml)
       │
       ▼
5-Replicate Configuration (configs/.../*_tuned_rep.yaml)
       │
       ▼
Evaluation Run Across Random Seeds (rep_0 -> rep_4)
```

1. **Exporting Best Parameters**:
   The best trial hyperparameters are saved to `configs/tuning/best/<study_name>_best.yaml`.
2. **Generating 5-Replicate Production Config**:
   `export_replicate_yaml` deep-copies the base pipeline template, merges the best parameters into the single pipeline definition, and generates a production multi-replicate configuration (`*_tuned_rep.yaml`) containing 5 replicates (`rep_0` to `rep_4`) with unique random seeds.
3. **Running the Replicates**:
   Execute the generated replicate config using the standard training runner to obtain statistical distributions (mean, std, interquartile ranges) across seeds:
   ```bash
   python scripts/cpc/train_primitive_rep.py --config configs/fr_cont/cpc/pr/sdsac_tuned_rep.yaml
   ```

---

## 5. Design Choices & Academic Rationales

### Evaluation Metric Selection: `mean_reward` over 100 Episodes

- **Metric**: `mean_reward` evaluated across `n_eval_episodes: 100`.
- **Rationale**:
  - **Avoids Plateauing & Gradient Saturation**: In continuous control and temporal logic tasks, binary `success_rate` suffers from severe saturation—it often remains `0.0` during early exploration and plateaus at `1.0` once the goal is reached. This produces flat optimization surfaces where Optuna cannot differentiate between mediocre and high-quality policies.
  - **Dense Guidance Signal**: `mean_reward` incorporates dense shaping signals (action penalties, distance-to-subgoal progress, time-to-completion, and collision penalties), steering the sampler toward policies that solve tasks efficiently and smoothly.
  - **Representation Learning Exception**: For unsupervised pretraining (such as ALLO's successor representation pretraining), the objective is set to `direction: minimize` with `metric: loss/total`. The pipeline utilizes `TrialLossCallback` to monitor training loss directly from the model logger without running environment rollouts.

### Tuning Budget Rationales

1. **Reduced Training Horizon (10%–25% of Full Steps)**:
   - *Academic Foundation*: **RL Baselines3 Zoo** (Raffin et al., JMLR 2021).
   - *Empirical Basis*: Extensive empirical benchmarking demonstrates that relative hyperparameter performance rankings establish early in training. Unpromising configurations (destabilizing learning rates, improper entropy scales) fail quickly and rarely recover. Training trials for 10%–25% of the full horizon provides accurate predictive signals at a fraction of the compute cost.
2. **Trial Count (25–40 Trials)**:
   - *Academic Foundation*: **TPE Convergence** (Bergstra et al., NeurIPS 2011).
   - *Empirical Basis*: The Tree-structured Parzen Estimator (TPE) algorithm models non-parametric densities $P(x|y)$ rather than joint distributions. For low- to moderate-dimensional search spaces (3–8 parameters), TPE reliably identifies near-optimal regions within 25–40 trials when initialized with 5 random startup trials (`n_startup_trials: 5`).
3. **Median Pruning**:
   - *Empirical Basis*: Optuna's `MedianPruner` compares intermediate evaluation checkpoints against the median performance of previous trials at the same step. Discarding the bottom 50% of trials early saves 50%–70% of total GPU compute time.
4. **Compute Allocation Principle**:
   - *Academic Foundation*: **Statistical RL Evaluation Standards** (Agarwal et al., NeurIPS 2021, *Deep RL at the Crossroads*).
   - *Principle*: Tuning compute should be budgeted conservatively, reserving the bulk of GPU resources for rigorous multi-seed evaluation (5+ replicates with distinct random seeds) to avoid hyperparameter overfitting on a single seed.

### Storage & Concurrency Architecture

- **Per-Experiment SQLite Databases (`db/<study_name>.db`)**:
  - **Concurrency Locks**: SQLite uses file-level locking during write operations. If multiple tuning processes write to a single centralized database file, parallel worker processes can trigger `sqlite3.OperationalError: database is locked`. Using independent SQLite database files for each experiment completely eliminates write lock contention.
  - **Failure Isolation**: Any unexpected process interruption or corrupt database state remains isolated to that specific study without affecting other ongoing experiments.
  - **Clean Dashboard Inspection**: Directing `optuna-dashboard` to a specific study database presents a clean, responsive view tailored to that specific experiment.

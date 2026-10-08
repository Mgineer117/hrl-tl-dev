# RL Reward Convergence Analysis

This directory contains comprehensive scripts to analyze the convergence of RL training from saved reward data and specification models. The tools provide detailed convergence metrics, visualizations, and automated analysis across multiple models.

## Overview

The convergence analysis toolkit consists of two main components:

1. **Individual Model Analysis** - Detailed analysis of single reward files with comprehensive metrics and visualizations
2. **Specification-based Analysis** - Batch analysis across multiple specification models with summarized results

## Files

1. **`convergence_analysis.py`** - Core convergence analysis for individual reward files
2. **`convergence_analysis_specs.py`** - Main analysis script for specification models
3. **`run_convergence_analysis.py`** - Example script to run analysis with different configurations  
4. **`visualize_convergence.py`** - Script to create visualizations and summaries from results
5. **`README.md`** - This documentation file

## Features

### Convergence Metrics
- **Final Mean Return**: Average performance in recent episodes
- **Trend Analysis**: Linear regression to measure improvement rate
- **Stability Measures**: Coefficient of variation and variance ratios
- **Convergence Detection**: Automatic detection of convergence point

### Visualizations
- Raw returns with smoothed trends
- Moving averages with different window sizes
- Rolling coefficient of variation
- Return distribution histograms

## Usage

### Individual Model Analysis

For analyzing single reward files with detailed metrics and visualizations:

#### Basic Usage
```bash
python convergence_analysis.py --file path/to/evaluations.npz
```

#### Command Line Options
- `--file`: Path to npz file containing returns (default: preset path)
- `--threshold`: Convergence detection threshold (default: 0.1)
- `--window`: Window size for convergence detection (default: 50)
- `--no-plot`: Skip generating plots
- `--save-plot`: Save plots to file

#### Examples

**Quick Analysis (no plots)**
```bash
python convergence_analysis.py --file data.npz --no-plot
```

**Strict Convergence Detection**
```bash
python convergence_analysis.py --file data.npz --threshold 0.05
```

**Large Window Analysis**
```bash
python convergence_analysis.py --file data.npz --window 100
```

**Generate and Save Plots**
```bash
python convergence_analysis.py --file data.npz --save-plot
```

### Specification Model Analysis

For batch analysis across multiple specification models:

#### Basic Usage

Run the convergence analysis for all specifications:

```bash
python convergence_analysis_specs.py
```

This will:
- Load specifications from `assets/formulae/fourroom/all_formulae_1_cla_1_max_pred.json`
- Look for evaluation files at `out/fourroom/ltl_ll/ll_policies/fourroom_tl_ppo_rand_{spec}/eval/evaluations.npz`
- Analyze convergence for each model that has evaluation data
- Save results to `convergence_summary.json`

#### Advanced Usage

You can customize the analysis by calling the function directly:

```python
from convergence_analysis_specs import check_convergence_for_all_specs

check_convergence_for_all_specs(
    formulae_file_path="assets/formulae/fourroom/all_formulae_1_cla_1_max_pred.json",
    base_eval_path="out/fourroom/ltl_ll/ll_policies",
    model_prefix="fourroom_tl_ppo_rand",  # or "fourroom_tl_ppo_stay"
    output_file="my_convergence_results.json",
    convergence_threshold=0.1,  # CV threshold for convergence
    convergence_window=50,  # Window size for analysis
)
```

#### Running with Different Model Prefixes

Use the example script to analyze both `fourroom_tl_ppo_rand` and `fourroom_tl_ppo_stay` models:

```bash
python run_convergence_analysis.py
```

#### Visualizing Results

After running the analysis, create visualizations:

```bash
python visualize_convergence.py
```

This will generate:
- PNG plots showing convergence statistics
- CSV tables with detailed results
- Console output with top performing models

## Data Format

The scripts expect npz files with a "returns" array:
- **1D array**: Direct sequence of returns over time
- **2D array**: (evaluation_steps, episodes) - will take mean across episodes

## File Path Structure

The specification analysis script expects evaluation files to be located at:
```
{base_eval_path}/{model_prefix}_{sanitized_spec_name}/eval/evaluations.npz
```

For example:
```
out/fourroom/ltl_ll/ll_policies/fourroom_tl_ppo_rand_Gpsi_gl/eval/evaluations.npz
```

The specification names are sanitized using `replace_special_characters()` from `gym_tl_tools`.

## Convergence Criteria

**Converged**: When coefficient of variation in a sliding window falls below threshold
- Default threshold: 0.1 (10% variation)
- Default window: 50 episodes
- Lower threshold = stricter convergence requirement

A model is considered converged when:
1. The coefficient of variation (CV) in a sliding window falls below the threshold (default: 0.1)
2. The CV remains below threshold for the specified window size (default: 50 episodes)

The CV is calculated as: `std(returns) / abs(mean(returns))`

## Output Format

### Individual Analysis Output

#### Convergence Status
- ✅ **CONVERGED**: Algorithm reached stable performance
- ❌ **NOT CONVERGED**: Still improving or unstable

#### Key Metrics
- **Trend Slope**: Positive = improving, negative = degrading
- **R² Value**: How linear the improvement is (higher = more consistent)
- **Coefficient of Variation**: Stability measure (lower = more stable)
- **Variance Ratio**: Late vs early stability (< 1.0 = more stable later)

#### Plots Generated
1. **Raw Returns**: Shows actual performance over time
2. **Moving Averages**: Smoothed trends with different window sizes
3. **Rolling CV**: Stability measure over time
4. **Distribution**: Histogram of return values

### Specification Analysis Output

#### JSON Output Structure

```json
{
  "summary": {
    "total_specifications": 196,
    "analyzed": 150,
    "converged": 120, 
    "not_converged": 30,
    "missing_evaluations": 46,
    "convergence_threshold": 0.1,
    "convergence_window": 50
  },
  "specifications": {
    "Gpsi_gl": {
      "spec_id": 0,
      "specification": "Gpsi_gl",
      "model_name": "fourroom_tl_ppo_rand_Gpsi_gl",
      "has_converged": true,
      "convergence_point": 1250,
      "total_episodes": 2000,
      "final_mean": 0.85,
      "final_std": 0.12,
      "recent_cv": 0.08,
      "trend_slope": 0.0001,
      "trend_r_squared": 0.95,
      "variance_ratio": 0.75
    }
  },
  "converged_specs": [...],
  "not_converged_specs": [...],
  "missing_evaluations": [...]
}
```

#### Key Metrics Explained

- **has_converged**: Whether the model has converged based on CV threshold
- **convergence_point**: Episode where convergence was detected
- **final_mean/std**: Performance in the final window
- **recent_cv**: Coefficient of variation in recent window (lower = more stable)
- **trend_slope**: Learning trend (positive = improving)
- **trend_r_squared**: How well the trend fits (higher = more consistent learning)
- **variance_ratio**: Stability comparison between early and late training

## Tips and Best Practices

### Individual Analysis Tips
- Use `--no-plot` for quick analysis or batch processing
- Adjust `--threshold` based on your stability requirements
- Larger `--window` sizes give more conservative convergence detection
- Save plots with `--save-plot` for reports and documentation

### Specification Analysis Tips
- Process subsets of specifications for very large datasets to avoid memory issues
- Use different model prefixes to compare training strategies
- Visualize results after analysis for better insights

## Dependencies

- `gym_tl_tools` (for `replace_special_characters`)
- `hrl_tl.eval.conv` (for `analyze_convergence`)
- `matplotlib` (for visualizations)
- `pandas` (for data processing)
- `numpy` (for numerical computations)
- Standard library: `json`, `os`, `typing`, `argparse`

## Troubleshooting

### No evaluation files found
- Check that the `base_eval_path` is correct
- Verify the `model_prefix` matches your model naming convention
- Ensure evaluation has been run for the models

### Import errors
- Make sure you're in the correct environment with all dependencies installed
- Check that the `hrl_tl` package is in your Python path

### Memory issues with large datasets
- The script loads evaluation data into memory
- For very large datasets, consider processing subsets of specifications

### File format issues
- Ensure npz files contain a "returns" array
- Check that return data is in the expected format (1D or 2D arrays)
- Verify file paths are correct and files are accessible

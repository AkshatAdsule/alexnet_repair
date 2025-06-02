# AlexNet Repair

This repository contains tools for repairing AlexNet models using constraint-based methods and studying the effects of different repair configurations.

## Basic Usage

1. **Generate Edit Sets**:
   ```bash
   # Generate misclassified edit set
   uv run python editset_generator.py misclassified

   # Generate class-specific edit set for cats (class 3)
   uv run python editset_generator.py by-class --target-class 3
   ```

2. **Repair Model**:
   ```bash
   # Repair on misclassified edit set
   uv run python run_repair.py misclassified

   # Repair on class-specific edit set
   uv run python run_repair.py by-class --target-class 3
   ```

3. **Evaluate Model**:
   ```bash
   # Evaluate base model
   uv run python eval.py --weights artifacts/alexnet_base.pth

   # Evaluate repaired model
   uv run python eval.py --weights artifacts/alexnet_repaired.pth
   ```

## Experiment Runner

The experiment runner automates the process of running multiple model repair experiments with varying configurations. This is useful for studying the impact of repair-set size and type on the repair process.

### Features

The experiment runner can automatically:

- Generate different types and sizes of edit sets
- Run model repairs with various parameters
- Measure repair performance (runtime, Gurobi iterations)
- Evaluate model accuracy before and after repair
- Calculate "drawdown" (accuracy impact on non-edit samples)
- Store all results in machine-readable CSV format

### Usage

#### Basic Experiment

```bash
# Simple experiment with misclassified edit sets
uv run python experiment_runner.py \
  --repair-set-sizes 10 50 100 \
  --repair-set-types misclassified \
  --output-file results/basic_experiment.csv
```

#### Class-Specific Experiments

```bash
# Experiment with class-specific edit sets
uv run python experiment_runner.py \
  --repair-set-sizes 25 50 100 \
  --repair-set-types class_homogeneous_incorrect \
  --target-classes 0 3 5 7 \
  --output-file results/class_experiment.csv
```

#### Full Experiment Suite

```bash
# Comprehensive experiment comparing different repair set types and sizes
uv run python experiment_runner.py \
  --repair-set-sizes 10 25 50 100 200 \
  --repair-set-types misclassified class_homogeneous_incorrect \
  --target-classes 0 1 2 3 4 5 6 7 8 9 \
  --param-bound 5.0 \
  --margin 2.0 \
  --output-file results/full_experiment.csv
```

### Available Options

- `--repair-set-sizes`: List of edit set sizes to test (e.g., `10 50 100`)
- `--repair-set-types`: Types of repair sets:
  - `misclassified`: Images the baseline model misclassifies
  - `class_homogeneous_correct`: Correctly classified images from a single class  
  - `class_homogeneous_incorrect`: Incorrectly classified images from a single class
- `--target-classes`: CIFAR-10 classes to test (0-9, default: all classes)
- `--param-bound`: Parameter change bound for repair (default: 5.0)
- `--margin`: Classification margin for repair (default: 2.0)
- `--output-file`: CSV file for results (default: `results/experiment_data.csv`)

### Output Format

The experiment runner produces a CSV file with the following columns:

#### Experiment Metadata
- `experiment_id`: Unique identifier for each run
- `timestamp`: When the experiment was executed

#### Configuration Parameters
- `repair_set_type`: Type of edit set used
- `repair_set_focus`: Classification focus (currently `incorrect_only`)
- `requested_repair_set_size`: Target size for edit set
- `edit_set_sizing_strategy`: Sizing strategy (currently `flexible`)
- `actual_repair_set_size`: Actual number of images in edit set
- `edit_set_path`: Path to the generated edit set file
- `target_class`: Target class (for class-specific experiments)
- `param_bound`: Parameter bound used for repair
- `margin`: Margin used for repair

#### Baseline Model Performance
- `baseline_model_path`: Path to baseline model
- `baseline_accuracy_test_set`: Baseline accuracy on full test set
- `baseline_accuracy_edit_set`: Baseline accuracy on edit set

#### Repair Process Metrics
- `repair_runtime_seconds`: Wall-clock time for repair process
- `gurobi_barrier_iterations`: Number of Gurobi solver iterations
- `repair_successful`: Whether repair succeeded (True/False)

#### Repaired Model Performance
- `repaired_model_path`: Path to repaired model (if successful)
- `repaired_accuracy_test_set`: Repaired model accuracy on full test set
- `repaired_accuracy_edit_set`: Repaired model accuracy on edit set
- `repaired_accuracy_drawdown_set`: Repaired model accuracy on test set excluding edit samples

#### Error Handling
- `error_message`: Error details if experiment failed

### Example Analysis

After running experiments, you can analyze the results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/experiment_data.csv')

# Plot repair set size vs. accuracy
plt.figure(figsize=(10, 6))
for repair_type in df['repair_set_type'].unique():
    subset = df[df['repair_set_type'] == repair_type]
    plt.plot(subset['requested_repair_set_size'], 
             subset['repaired_accuracy_test_set'], 
             marker='o', label=repair_type)

plt.xlabel('Repair Set Size')
plt.ylabel('Test Set Accuracy After Repair')
plt.legend()
plt.title('Repair Set Size vs. Model Performance')
plt.show()

# Analyze runtime vs. set size
plt.figure(figsize=(10, 6))
plt.scatter(df['actual_repair_set_size'], df['repair_runtime_seconds'])
plt.xlabel('Edit Set Size')
plt.ylabel('Repair Runtime (seconds)')
plt.title('Repair Runtime vs. Edit Set Size')
plt.show()
```

## Project Structure

```
alexnet_repair/
├── artifacts/              # Model files and outputs
├── data/                    # CIFAR-10 data and edit sets
├── edits/                   # Repair algorithms
├── editset_helpers/         # Edit set generation utilities
├── helpers/                 # General utilities
├── model_classes/           # Model definitions
├── sytorch/                 # Symbolic computation framework
├── templates/               # Web interface templates
├── editset_generator.py     # Generate different edit sets
├── experiment_runner.py     # Automated experiment execution
├── run_repair.py           # Run model repair
├── eval.py                 # Model evaluation
└── README.md               # This file
```

## CIFAR-10 Classes

```
0: airplane    1: automobile  2: bird     3: cat      4: deer
5: dog         6: frog        7: horse    8: ship     9: truck
```

## Development

This project uses UV for dependency management. The design specification for the experiment runner can be found in `EXPERIMENT_RUNNER_DESIGN_SPEC.md`.

For more detailed usage instructions, run any script with `--help`:

```bash
uv run python experiment_runner.py --help
uv run python editset_generator.py --help  
uv run python run_repair.py --help
```

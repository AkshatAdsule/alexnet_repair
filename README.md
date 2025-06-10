# AlexNet Repair

A toolkit for repairing CIFAR-10 AlexNet models using APRNN, and for studying the impact of different repair configurations on model performance.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Command-Line Tools](#command-line-tools)
   - [Edit Set Generation](#1-edit-set-generation)
   - [Model Repair](#2-model-repair)
   - [Model Evaluation](#3-model-evaluation)
   - [Experiment Runner](#4-experiment-runner)
   - [Utilities & Analysis](#5-utilities--analysis)
7. [Directory Structure](#directory-structure)
8. [CIFAR-10 Classes](#cifar-10-classes)
9. [Development & Contribution](#development--contribution)
10. [Citation](#citation)
11. [License](#license)


## Overview

AlexNet Repair is a Python-based framework for:

- Generating **edit sets** (patches of training or test samples) highlighting model misbehavior.
- Applying **constraint-based repairs** to the AlexNet architecture using a Gurobi backend.
- Evaluating and analyzing the effects of repairs on overall model accuracy and on specific classes.

This project leverages the framework `sytorch` to express and solve repair constraints.


## Features

- Flexible edit set generation (misclassified samples, class-specific).
- Parameter-bound and margin-based repairs via mixed-integer programming.
- Automated experiments with variable repair-set sizes and types.
- Detailed logging of solver metrics (runtime, Gurobi iterations).
- Post-repair evaluation and drawdown analysis.
- Utilities for stochastic analysis, model conversion, and verification.


## Prerequisites

- **Python** 3.8 or higher
- **Gurobi** optimizer (version 9.0+) with a valid license
- **CIFAR-10** dataset (automatically downloaded `data/`)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-org/alexnet_repair.git
    cd alexnet_repair
    ```

2. Install dependencies using UV (the project's dependency manager):
    ```bash
    uv sync
    ```
   or via pip:
    ```bash
    pip install -r requirements.txt
    ```

## Quickstart

### 1. Generate an Edit Set

- Misclassified samples:
  ```bash
  uv run python editset_generator.py misclassified --max-size 100
  ```

- Class-specific incorrect samples (e.g., class 3 = "cat"):
  ```bash
  uv run python editset_generator.py by-class --target-class 3 --max-size 50
  ```

Generated sets are saved under `data/edit_sets/`.


### 2. Repair the Model

- Repair on a misclassified edit set:
  ```bash
  uv run python run_repair.py misclassified --param-bound 5.0 --margin 2.0
  ```

- Repair on a class-specific edit set:
  ```bash
  uv run python run_repair.py by-class --target-class 3 --param-bound 3.0 --margin 1.5
  ```

The repaired weights are saved in `artifacts/alexnet_repaired.pth`.


### 3. Evaluate Model Performance

- Evaluate baseline AlexNet:
  ```bash
  uv run python eval.py --weights artifacts/alexnet_base.pth --cuda
  ```

- Evaluate repaired AlexNet:
  ```bash
  uv run python eval.py --weights artifacts/alexnet_repaired.pth --cuda
  ```

Results are printed to console and logged in `results/`.


## Command-Line Tools

### 1. Edit Set Generation

**Script:** `editset_generator.py`

Usage:
```bash
uv run python editset_generator.py <mode> [options]
```

- `<mode>`:
  - `misclassified`: images the base model misclassifies
  - `by-class`: samples from a single class (correct or incorrect)

- **Options**:
  - `--max-size`: Maximum number of samples
  - `--target-class`: (for `by-class` mode) integer 0–9

### 2. Model Repair

**Script:** `run_repair.py`

Usage:
```bash
uv run python run_repair.py <mode> [options]
```

- **Options**:
  - `--param-bound`: maximum allowed parameter change (default: 5.0)
  - `--margin`: classification margin (default: 2.0)
  - `--output`: path to save repaired model (default: `artifacts/alexnet_repaired.pth`)

### 3. Model Evaluation

**Script:** `eval.py`

Usage:
```bash
uv run python eval.py --weights <model_path> [--cuda] [--batch-size N]
```

- `--weights`: path to `.pth` model file
- `--cuda`: enable GPU evaluation
- `--batch-size`: evaluation batch size (default: 128)


### 4. Experiment Runner

**Script:** `experiment_runner.py`

Automates generating edit sets, repairing, and evaluating over multiple configurations.

Usage:
```bash
uv run python experiment_runner.py --repair-set-sizes 10 50 100 \
    --repair-set-types misclassified class_homogeneous_incorrect \
    --target-classes 0 1 2 3 \
    --param-bound 5.0 --margin 2.0 \
    --output-file results/experiment.csv
```


### 5. Utilities & Analysis

- `analysis.py` / `analyze_stochastic.py`: scripts for analysis and visualization.

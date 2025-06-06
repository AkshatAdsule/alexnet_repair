---
applyTo: "**"
---

# Python UV Package Manager Rule

This project uses `uv` as the Python package manager. Always use `uv` commands for Python operations.

## Key Files

- [pyproject.toml](mdc:pyproject.toml) - Project configuration and dependencies
- [uv.lock](mdc:uv.lock) - Locked dependency versions

## Required Commands

### Running Python Scripts

- **Always use**: `uv run script_name.py`
- **Never use**: `python script_name.py` or `python3 script_name.py`

### Installing Dependencies

- **Always use**: `uv add package_name`
- **Never use**: `pip install package_name`

### Installing Development Dependencies

- **Always use**: `uv add --dev package_name`
- **Never use**: `pip install package_name`

### Running with Specific Python Files

- **Always use**: `uv run python main.py`
- **Always use**: `uv run python edit_set_visualizer.py`
- **Always use**: `uv run python experiments.py`

### Installing from Requirements

- **Always use**: `uv sync` (to sync from uv.lock)
- **Never use**: `pip install -r requirements.txt`

### Virtual Environment Management

- `uv` automatically manages virtual environments
- **Never activate/deactivate manually** - `uv run` handles this

### Common Project Commands

- Start the visualizer: `uv run python edit_set_visualizer.py`
- Run experiments: `uv run python experiments.py`
- Run main script: `uv run python main.py`
- Run tests: `uv run python test_experiments.py`

## Why UV?

- Faster dependency resolution and installation
- Better dependency management with uv.lock
- Automatic virtual environment handling
- Native support for pyproject.toml

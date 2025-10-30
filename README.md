# CQT-experiments

This project provides a benchmarking suite for quantum experiments. The system uses a batch runner (`scripts_executor`) that executes experiments defined in an ini configuration file, organizing results in a standardized directory structure.

To test your software, you have to call

`./run_sinq20.sh`

## Project Structure

```
CQT-experiments/
├── scripts/
│   ├── <experiment1>/
│   │   └── main.py          # Experiment entry point
│   ├── config.py            # Helper utilities for path resolution
│   └── scripts_executor.py  # Batch runner that reads ini file
├── data/                    # where the data gets stored
├── experiments_list.ini     # Defines which experiments to run
└── pyproject.toml
|__ upload.py.               # uploads the results of the experiments in the database
```

## How It Works

The `scripts_executor` reads the `experiments_list.ini` file to determine which experiments to run. Each experiment is organized in sections within the ini file:

- **[calibration]** - Core calibration experiments (DO NOT COMMENT)
- **[i]** - Subroutine that are using i qubits.

The executor runs each enabled experiment's `main.py` with the specified parameters, collecting results in the standardized `data/` structure.

## Development

Perform this in your account (is creating a virtualenv with libraries that are not installed in the default qibo module)

```
  module load qibo
  python -m venv ~/envs/qibo_env
  source ~/envs/qibo_env/bin/activate
  pip install --upgrade pip
  pip install GitPython
  pip install -e .  # Install project and all dependencies
```

## Rules for Adding New Experiments

- **Package Dependencies**: Add new packages to `pyproject.toml` (dependencies), e.g. torch, quiboml, quibocal
- The **Directory** of your experiment must be `scripts/<experiment>/`
- The **Entry Point** of your script must be `scripts/<experiment>/main.py`
- The **output** of your script must be  obtained as follows:

```python
from pathlib import Path
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); import config

# Resolve paths
out_dir = config.output_dir_for(__file__) / args.device
out_dir.mkdir(parents=True, exist_ok=True)
```


- **Device Support**: `device` is provided via `--device` and must be either `numpy` or `nqch`
- **Directory Creation**: Use `mkdir(parents=True, exist_ok=True)` before writing results
- Other arguments are `--qubits_list`, which is a list of edges, where the edge is a list of length 2: `"[[0, 1], [0, 3], [1, 4], [2, 3]"`

### Additional Guidelines

- **Arguments**: Provide `argparse` with sensible defaults; do not override CLI args in code
- **Integration**: Ensure the script runs when added to the ini configuration file
- **Artifacts**: Optional extra artifacts (plots, params, matrices) go in `data/<experiment>/<device>/...`
- **Histograms**: For histogram-like results, include all bitstrings (zero for missing) in frequencies dict
- **Documentation**: Document that `"numpy"` is supported for local simulation

### Configuration File

Add your experiment to the appropriate section in `config.ini`. The `[calibration]` section contains core experiments that should remain uncommented, while other sections can be selectively enabled/disabled by commenting out experiment entries.
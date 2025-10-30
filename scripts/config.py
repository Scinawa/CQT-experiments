from pathlib import Path
from git.repo.base import Repo
import logging
import configparser

# Repository root (two levels above any script in scripts/<name>/main.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

CURRENT_CALIBRATION_DIRECTORY = "/mnt/scratch/qibolab_platforms_nqch"
RUN_ID_FILE = REPO_ROOT / "current_run_id.json"



def output_dir_for(script_file: str, device: str | Path) -> Path:
    """Return data/<script-dir-name>/ for the given script file."""
    script_path = Path(script_file).resolve()

    if device == "numpy":
        return DATA_DIR / script_path.parent.name / device
    else:

        repo = Repo(CURRENT_CALIBRATION_DIRECTORY)
        hash_id = repo.commit().hexsha

        return DATA_DIR / script_path.parent.name / hash_id



def load_experiment_list(config_file="experiment_list.ini", logger=None):
    """
    Load experiment list from an INI configuration file.

    Args:
        config_file (str): Path to the experiment list INI configuration file

    Returns:
        dict: Dictionary with sections as keys and lists of experiments as values
    """
    experiments = {}
    try:
        config = configparser.ConfigParser()
        config.read(config_file)

        for section in config.sections():
            experiments[section] = []
            for key, value in config[section].items():
                # Only include experiments that are enabled (not commented out)
                if not key.startswith("#") and value.lower() in [
                    "enabled",
                    "true",
                    "1",
                ]:
                    experiments[section].append(key)

    except Exception as e:
        logger.error(f"Error reading experiment list from '{config_file}': {e}")
        return {}
    return experiments
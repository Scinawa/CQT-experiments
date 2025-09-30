import os
import sys
import subprocess
import argparse
import logging
from logging.handlers import RotatingFileHandler
import shutil
from git.repo.base import Repo
from datetime import datetime
import sys
from pathlib import Path
import json

# Base path to the scripts directory (run from project root)
base_path = "scripts/"


if __package__ is None or __package__ == "":
    # invoked directly: add repo root to sys.path so 'scripts.*' resolves
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from scripts.config import CURRENT_CALIBRATION_DIRECTORY  # now this works in both cases



def load_experiment_list(config_file="scripts/experiment_list.txt"):
    """
    Load experiment list from a configuration file.

    Args:
        config_file (str): Path to the experiment list configuration file

    Returns:
        list: List of experiment names (uncommented lines)
    """
    experiments = []
    try:
        with open(config_file, "r") as f:
            for line in f:
                # Strip whitespace and skip empty lines
                line = line.strip()
                if not line:
                    continue
                # Skip comment lines (starting with #)
                if line.startswith("#"):
                    continue
                # Add the experiment name
                experiments.append(line)
    except FileNotFoundError:
        logging.warning(
            f"Experiment list file '{config_file}' not found."
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading experiment list from '{config_file}': {e}")
    return experiments


# Load experiment list from configuration file
experiment_list = load_experiment_list()


def parse_args():
    parser = argparse.ArgumentParser(description="Run all experiment scripts.")
    parser.add_argument(
        "--device",
        choices=["numpy", "sinq20"],
        default="numpy",
        help="Execution device to pass to each experiment script.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/runscripts.log",
        help="Path to the log file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=experiment_list,
        help="List of experiment subfolders to run.",
    )
    return parser.parse_args()


def setup_logger(log_file: str, level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger("runscripts")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)

    fh = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def run_script(logger: logging.Logger, script_path: str, device: str, tag: str) -> int:
    if not os.path.exists(script_path):
        logger.warning(f"main.py not found in {tag}")
        return 1

    logger.info(f"Running {script_path} with device={device}")
    cmd = [sys.executable, "-u", script_path, "--device", device]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.info(f"[{tag}] {line.rstrip()}")
        proc.wait()
        if proc.returncode != 0:
            logger.error(f"{script_path} exited with code {proc.returncode}")
        else:
            logger.info(f"Finished {script_path}")
        return proc.returncode or 0
    except Exception:
        logger.exception(f"Error occurred while running {script_path}")
        return 1

def copytree_safe(src: Path, dst: Path, ignore_dirs=None):
    """
    Recursively copy directory `src` into `dst`, skipping directories in `ignore_dirs`
    and continuing on permission (or other) errors. Creates directories as needed.
    """
    src = Path(src)
    dst = Path(dst)
    ignore_dirs = set(ignore_dirs or [])

    for root, dirs, files in os.walk(src, topdown=True):
        root_path = Path(root)

        # prune ignored directories in-place so os.walk won't descend into them
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        # compute destination folder corresponding to current root
        rel = root_path.relative_to(src)
        dest_root = dst / rel
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {dest_root}: {e}")
            # continue anyway; file copies may still succeed for siblings
            pass

        # copy files under this root
        for name in files:
            src_file = root_path / name
            dest_file = dest_root / name
            try:
                # create parent in case mkdir above failed
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
            except Exception as e:
                logger.warning(f"Skipping {src_file} -> {dest_file}: {e}")


def main():
    args = parse_args()
    logger = setup_logger(args.log_file, args.log_level)

    # COPY RUNCARD INTO DATA SECTION
    repo = Repo(CURRENT_CALIBRATION_DIRECTORY)
    commit = repo.commit()
    hash_id = commit.hexsha

    # Copy /mnt/scratch/qibolab_platforms_nqch into data/<hash_id>/
    calibration_dir = os.path.join("data", hash_id)


    try:
        # Copy, skipping .git and continuing on any copy errors
        copytree_safe(
            Path(CURRENT_CALIBRATION_DIRECTORY),
            calibration_dir,
            ignore_dirs={".git", "__pycache__"},
        )

        # Prepare JSON content with custom format
        time_format = "%d-%m-%Y %H:%M"

        commit_info = {
            "commit_hash": hash_id,
            "commit_message": commit.message.strip(),
            "experiment_date": datetime.now().strftime(time_format),
            "calibration_date": datetime.fromtimestamp(commit.committed_date).strftime(time_format),
        }

        # Write commit info to JSON file
        msg_file = os.path.join(calibration_dir, "commit_info.json")
        with open(msg_file, "w") as f:
            json.dump(commit_info, f, indent=4)

    except Exception as e:
        logger.error(f"Failed to copy calibration directory: {e}")


    # Remove the .git directory inside the copied calibration directory via shell
    git_dir = os.path.join(calibration_dir, ".git")
    if os.path.isdir(git_dir):
        logger.info(f"Removing git directory {git_dir}")
        try:
            subprocess.run(f"rm -rf -- '{git_dir}'", shell=True, check=True)
        except subprocess.CalledProcessError:
            logger.error(f"Shell removal failed for {git_dir}; attempting fallback")
            try:
                shutil.rmtree(git_dir)
            except Exception:
                logger.exception(f"Fallback removal failed for {git_dir}")

    overall_rc = 0
    for subfolder in args.experiments:
        script_path = os.path.join(base_path, subfolder, "main.py")
        print("\n\n\n")
        rc = run_script(logger, script_path, args.device, subfolder)
        overall_rc = overall_rc or rc  # keep first non-zero

    sys.exit(overall_rc)


if __name__ == "__main__":
    main()

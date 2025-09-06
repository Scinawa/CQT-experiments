import os
import sys
import subprocess
import argparse
import logging
from logging.handlers import RotatingFileHandler
import shutil
from git.repo.base import Repo

 


# Base path to the scripts directory (run from project root)
base_path = "scripts/"
experiment_list = [
#       "GHZ",
#          "mermin",
#     "grover2q",  # very broken (see above)
# #    "tomography", # very broken
# #    "process_tomography", # very broken (see above)
#    "grover3q",
#     "universal_approximant",
# #     "reuploading_classifier",
#     "QFT",
#     "qml_3Q_yeast",
    "qml_4Q_yeast",
    "qml_3Q_statlog",
    "qml_4Q_statlog"
]


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


def main():
    args = parse_args()
    logger = setup_logger(args.log_file, args.log_level)


    # COPY RUNCARD INTO DATA SECTION
    repo = Repo("/mnt/scratch/qibolab_platforms_nqch")
    hash_id = repo.commit().hexsha

    # Copy /mnt/scratch/qibolab_platforms_nqch into 
    # data/<hash_id>/
    runcard_dir = os.path.join("data", hash_id)

    try:
        shutil.copytree("/mnt/scratch/qibolab_platforms_nqch", runcard_dir, dirs_exist_ok=True)
    except Exception as e:
        #print("diocas")
        logger.error(f"Failed to copy runcard directory")
        # sys.exit(1)



    overall_rc = 0
    for subfolder in args.experiments:
        script_path = os.path.join(base_path, subfolder, "main.py")
        print("\n\n\n")
        rc = run_script(logger, script_path, args.device, subfolder)
        overall_rc = overall_rc or rc  # keep first non-zero

    sys.exit(overall_rc)


if __name__ == "__main__":
    main()

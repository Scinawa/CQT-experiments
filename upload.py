"""
Clean, structured script for uploading calibration and experiment data.
"""

import os
import json
import tomllib
import tarfile
import tempfile
import shutil
import logging
import argparse
import atexit
from pathlib import Path
from git import Repo

from scripts.config import RUN_ID_FILE, load_experiment_list
from utils import upload_calibration, upload_experiment


logger = logging.getLogger(__name__)


def setup_logging(log_level: str):
    """Setup logging configuration."""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/upload.log')
        ]
    )


def get_server_credentials(secrets_path: str = ".secrets.toml") -> tuple[str, str]:
    """
    Load server URL and API token from secrets file.
    
    Args:
        secrets_path: Path to the secrets TOML file.
    
    Returns:
        (server_url, api_token): Server credentials.
    
    Raises:
        ValueError: If credentials are missing.
        Exception: If file cannot be read.
    """
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    server_url = secrets.get("qibodbhost")
    api_token = secrets.get("qibodbkey")
    
    if not server_url or not api_token:
        raise ValueError("qibodbhost or qibodbkey missing in secrets file")
    
    return server_url, api_token


def get_run_id(path: str = RUN_ID_FILE) -> str:
    """
    Load run ID from JSON file.
    
    Args:
        path: Path to the run ID file.
    
    Returns:
        Run ID as a string.
    
    Raises:
        ValueError: If run_id is missing from the file.
        Exception: If file cannot be read.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    run_id = data.get("run_id") or data.get("experiment_id")
    if not run_id:
        raise ValueError("run_id missing from file")
    
    return str(run_id)


def prepare_calibration_directory(hash_id: str, calibration_source: str) -> str:
    """
    Prepare calibration directory. If hash_id is "latest", copy from source.
    
    Args:
        hash_id: Hash ID for the calibration (or "latest").
        calibration_source: Source directory to copy calibration from.
    
    Returns:
        (actual_hash_id, calibration_dir): Updated hash ID and directory path.
    """
    if hash_id == "latest":
        repo = Repo(calibration_source)
        actual_hash_id = repo.commit().hexsha
        calibration_dir = os.path.join("data", actual_hash_id)
        
        if not os.path.exists(calibration_dir):
            logger.info(f"Copying {calibration_source} to {calibration_dir}")
            shutil.copytree(calibration_source, calibration_dir)
        
        return actual_hash_id, calibration_dir
    else:
        return hash_id, f"./data/{hash_id}"


def compress_directory(src_dir: str, output_path: str) -> str:
    """
    Compress a directory into a tar.gz archive.
    
    Args:
        src_dir: Source directory to compress.
        output_path: Path for the output archive.
    
    Returns:
        Path to the created archive.
    
    Raises:
        FileNotFoundError: If source directory doesn't exist.
    """
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    logger.debug(f"Creating archive from {src_dir}")
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(src_dir, arcname=os.path.basename(src_dir))
    
    logger.debug(f"Created archive: {output_path}")
    return output_path


def upload_calibration_data(
    calibration_dir: str,
    hash_id: str,
    notes: str,
    server_url: str,
    api_token: str
):
    """
    Compress and upload calibration directory.
    
    Args:
        calibration_dir: Directory containing calibration data.
        hash_id: Hash ID for the calibration.
        notes: Notes for the upload.
        server_url: Server URL.
        api_token: API token.
    
    Raises:
        Exception: If compression or upload fails.
    """
    archive_name = f"{hash_id}.tar.gz"
    
    try:
        # Compress calibration directory
        compress_directory(calibration_dir, archive_name)
        
        # Upload to server
        logger.info(f"Uploading calibration with hash ID: {hash_id}")
        upload_calibration(
            hash_id=hash_id,
            archive_path=archive_name,
            notes=notes,
            server_url=server_url,
            api_token=api_token
        )
        logger.info("Calibration upload completed successfully")
    finally:
        # Clean up archive file
        if os.path.exists(archive_name):
            try:
                os.remove(archive_name)
                logger.debug(f"Cleaned up archive: {archive_name}")
            except OSError as e:
                logger.warning(f"Failed to cleanup archive {archive_name}: {e}")


def upload_experiment_data(
    experiment_name: str,
    experiment_dir: str,
    hash_id: str,
    run_id: str,
    server_url: str,
    api_token: str
) -> bool:
    """
    Compress and upload a single experiment.
    
    Args:
        experiment_name: Name of the experiment.
        experiment_dir: Directory containing experiment data.
        hash_id: Hash ID for the experiment.
        run_id: Run ID for the experiment.
        server_url: Server URL.
        api_token: API token.
    
    Returns:
        True if successful, False otherwise.
    """
    if not os.path.exists(experiment_dir):
        logger.warning(f"Experiment directory not found: {experiment_dir}")
        return False
    
    logger.debug(f"Found experiment directory: {experiment_dir}")
    
    # Create archive in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_name = f"{experiment_name}_{hash_id}_{run_id}.tar.gz"
        temp_archive_path = os.path.join(tmpdir, archive_name)
        
        logger.debug(f"Creating archive: {archive_name}")
        compress_directory(experiment_dir, temp_archive_path)
        
        # Move archive to current directory for upload
        final_archive_path = os.path.abspath(archive_name)
        shutil.move(temp_archive_path, final_archive_path)
    
    # Register cleanup after program exit
    def cleanup():
        if os.path.exists(final_archive_path):
            try:
                os.remove(final_archive_path)
                logger.debug(f"Cleaned up archive: {final_archive_path}")
            except OSError as e:
                logger.warning(f"Failed to cleanup archive {final_archive_path}: {e}")
    
    atexit.register(cleanup)
    
    # Upload to server
    try:
        logger.info(f"Uploading {experiment_name} results")
        upload_experiment(
            hash_id=hash_id,
            name=experiment_name,
            archive_path=final_archive_path,
            experiment_id=run_id,
            notes=f"Results for {experiment_name}",
            server_url=server_url,
            api_token=api_token
        )
        logger.info(f"Successfully uploaded {experiment_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {experiment_name}: {e}")
        return False


def main():
    """Main upload workflow."""
    parser = argparse.ArgumentParser(description="Upload calibration data and experiment results")
    parser.add_argument("--hash-id", help="Hash ID for the calibration", default="latest")
    parser.add_argument("--notes", default="", help="Optional notes for the calibration upload")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set the logging level")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting upload process")
    
    # Get server credentials
    try:
        server_url, api_token = get_server_credentials()
        logger.info("Loaded server credentials")
    except Exception as e:
        logger.error(f"Failed to load server credentials: {e}")
        return 1
    
    # Get experiment groups and run ID
    try:
        experiment_groups = load_experiment_list()
        run_id = get_run_id()
        logger.info(f"Loaded run ID: {run_id}")
    except Exception as e:
        logger.error(f"Failed to load experiment configuration: {e}")
        return 1
    
    # Prepare calibration directory
    try:
        hash_id, calibration_dir = prepare_calibration_directory(
            args.hash_id,
            "/mnt/scratch/qibolab_platforms_nqch"
        )
        logger.info(f"Using hash ID: {hash_id}")
    except Exception as e:
        logger.error(f"Failed to prepare calibration directory: {e}")
        return 1
    
    # Upload calibration data
    try:
        upload_calibration_data(
            calibration_dir=calibration_dir,
            hash_id=hash_id,
            notes=args.notes,
            server_url=server_url,
            api_token=api_token
        )
    except Exception as e:
        logger.error(f"Calibration upload failed: {e}")
        return 1
    
    # Flatten experiment list (exclude 'calibration' section)
    experiment_list = []
    for section_name, experiments in experiment_groups.items():
        if section_name.lower() != "calibration":
            experiment_list.extend(experiments)
    
    # Upload each experiment
    logger.info(f"Starting experiment results upload for {len(experiment_list)} experiments")
    successful_uploads = 0
    failed_uploads = 0
    
    for experiment_name in experiment_list:
        logger.debug(f"Processing experiment: {experiment_name}")
        experiment_dir = f"./data/{experiment_name}/{hash_id}/{run_id}"
        
        success = upload_experiment_data(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            hash_id=hash_id,
            run_id=run_id,
            server_url=server_url,
            api_token=api_token
        )
        
        if success:
            successful_uploads += 1
        else:
            failed_uploads += 1
    
    logger.info(f"Upload process completed! Successful: {successful_uploads}, Failed: {failed_uploads}")
    return 0 if failed_uploads == 0 else 1


if __name__ == "__main__":
    exit(main())

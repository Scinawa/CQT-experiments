import os
import tarfile
import argparse
import logging
import tempfile
import shutil
import atexit

from clientdb.client import (
    set_server,
    calibrations_upload, calibrations_list, calibrations_download, calibrations_get_latest,
    results_upload, results_download,unpack,test,results_list
)

# from scripts.scripts_executor import experiment_list



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
            f"Experiment list file '{config_file}' not found. Using fallback list."
        )
        # Fallback to original hardcoded list if file not found
        experiments = ["mermin"]
    except Exception as e:
        logging.error(f"Error reading experiment list from '{config_file}': {e}")
        experiments = ["mermin"]

    return experiments



def setup_logging(log_level):
    """Setup logging configuration"""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/upload.log')
        ]
    )


def upload_calibration_compressed(src_dir, hash_id, notes=""):
    """
    Compress a directory and upload it as calibration data.
    
    Args:
        src_dir (str): Path to the source directory to compress
        hash_id (str): Hash ID for the calibration
        notes (str): Optional notes for the upload
        
    Returns:
        str: The archive filename that was created
    """
    logger = logging.getLogger(__name__)
    archive_name = f"{hash_id}.tar.gz"
    
    if not os.path.isdir(src_dir):
        error_msg = f"Source directory not found: {src_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Creating archive from {src_dir}")
    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(src_dir, arcname=os.path.basename(src_dir))

    logger.info(f"Created archive: {archive_name}")

    try:
        calibrations_upload(hashID=hash_id, notes=notes, files=[archive_name])
        logger.info(f"Successfully uploaded calibration data with hash ID: {hash_id}")
    except Exception as e:
        logger.error(f"Failed to upload calibration data: {e}")
        raise
    finally:
        # Clean up the archive file
        try:
            os.remove(archive_name)
            logger.debug(f"Cleaned up archive: {archive_name}")
        except OSError as e:
            logger.warning(f"Failed to cleanup archive {archive_name}: {e}")
    
    return archive_name


def main():
    parser = argparse.ArgumentParser(description="Upload calibration data and experiment results")
    parser.add_argument("--hash-id", help="Hash ID for the calibration", default="9848c933bfcafbb8f81c940f504b893a2fa6ac23")
    parser.add_argument("--notes", default="", help="Optional notes for the calibration upload")
    parser.add_argument("--log-level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Set the logging level")
    args = parser.parse_args()

    # Setup logging
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting upload process")

    # Set up the server connection
    set_server("http://157.230.158.238", api_token="H1jqzIl7bFTvHHDR642QV0VgsF0KP5jJr8s2Vhy4OvE")
    logger.info("Connected to server")
    
    # Upload calibration data
    src_dir = f"./data/{args.hash_id}"
    logger.info(f"Uploading calibration data from {src_dir}")
    try:
        archive_name = upload_calibration_compressed(
            src_dir=src_dir,
            hash_id=args.hash_id,
            notes=args.notes
        )
        logger.info(f"Calibration upload completed successfully")
    except Exception as e:
        logger.error(f"Calibration upload failed: {e}")
        return 1
    

    experiment_list = load_experiment_list()

    # Upload results for each experiment
    logger.info(f"Starting experiment results upload for {len(experiment_list)} experiments")
    successful_uploads = 0
    failed_uploads = 0
    
    for experiment_name in experiment_list:
        logger.debug(f"Processing experiment: {experiment_name}")
        experiment_dir = f"./data/{experiment_name}/{args.hash_id}"
        
        if os.path.exists(experiment_dir):
            logger.info(f"Found experiment directory: {experiment_dir}")
            
            # Create a tar.gz archive of the entire experiment directory using a TemporaryDirectory
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_name = f"{experiment_name}_{args.hash_id}.tar.gz"
                temp_archive_path = os.path.join(tmpdir, archive_name)
                
                logger.debug(f"Creating archive: {archive_name}")
                with tarfile.open(temp_archive_path, "w:gz") as tar:
                    tar.add(experiment_dir, arcname=os.path.basename(experiment_dir))
                
                # Move archive to current working directory so it persists after temp dir cleanup
                final_archive_path = os.path.abspath(archive_name)
                shutil.move(temp_archive_path, final_archive_path)
                logger.debug(f"Archive created at: {final_archive_path}")

            # Register cleanup of the archive after program exit
            def _cleanup(path):
                try:
                    os.remove(path)
                    logger.debug(f"Cleaned up archive: {path}")
                except OSError as e:
                    logger.warning(f"Failed to cleanup archive {path}: {e}")
            atexit.register(_cleanup, final_archive_path)

            try:
                logger.info(f"Uploading {experiment_name} results")
                resp = results_upload(
                    hashID=args.hash_id,
                    name=experiment_name,
                    notes=f"Results for {experiment_name}",
                    files=[final_archive_path]
                )
                logger.info(f"Successfully uploaded {experiment_name}: {resp}")
                successful_uploads += 1
            except Exception as e:
                logger.error(f"Failed to upload {experiment_name}: {e}")
                failed_uploads += 1
        else:
            logger.warning(f"Experiment directory not found: {experiment_dir}")
            failed_uploads += 1
    
    logger.info(f"Upload process completed! Successful: {successful_uploads}, Failed: {failed_uploads}")
    return 0 if failed_uploads == 0 else 1


if __name__ == "__main__":
    main()
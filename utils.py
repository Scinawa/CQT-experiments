"""
Utility functions for interacting with the calibration/results API.
This module replaces the clientdb.client code with clean, direct API calls.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import requests


logger = logging.getLogger(__name__)


def _get_defaults(server_url: Optional[str], api_token: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Resolve server_url and api_token from arguments or environment/config.
    
    Args:
        server_url: Optional explicit server URL.
        api_token: Optional explicit token.
    
    Returns:
        (server_url, api_token): Final values.
    """
    return (
        server_url if server_url else os.getenv("QIBO_SERVER_URL", "http://127.0.0.1:5050"),
        api_token if api_token is not None else os.getenv("QIBO_API_TOKEN")
    )


def _auth_headers(api_token: Optional[str]) -> dict:
    """Build Authorization header dict if a token is provided."""
    return {"Authorization": f"Bearer {api_token}"} if api_token else {}


def upload_calibration(
    hash_id: str,
    archive_path: str,
    notes: str = "",
    server_url: Optional[str] = None,
    api_token: Optional[str] = None
) -> dict:
    """
    Upload a calibration archive to the server.
    
    Uploads a pre-compressed archive file to the server's /calibrations/upload endpoint.
    
    Args:
        hash_id: Unique identifier for the calibration record.
        archive_path: Path to the archive file (e.g., .tar.gz or .zip).
        notes: Optional notes for the calibration upload.
        server_url: Server URL (if None, uses default).
        api_token: API token (if None, uses default).
    
    Returns:
        dict: Server response containing status, id, and created_at.
    
    Raises:
        FileNotFoundError: If archive file doesn't exist.
        requests.HTTPError: If upload fails.
    """
    server_url, api_token = _get_defaults(server_url, api_token)
    
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    
    # Read the archive file
    with open(archive_path, 'rb') as f:
        archive_data = f.read()
    
    # Prepare upload
    files_payload = {"archive": (os.path.basename(archive_path), archive_data, "application/gzip")}
    data_payload = {"hashID": hash_id, "notes": notes or ""}
    url = server_url + "/calibrations/upload"
    
    logger.debug(f"Uploading calibration to {url}")
    resp = requests.post(
        url,
        data=data_payload,
        files=files_payload,
        headers=_auth_headers(api_token),
        timeout=300
    )
    
    if resp.status_code >= 400:
        raise requests.HTTPError(f"Calibration upload failed ({resp.status_code}): {resp.text}")
    
    return resp.json()


def upload_experiment(
    hash_id: str,
    name: str,
    archive_path: str,
    experiment_id: Optional[str] = None,
    notes: str = "",
    server_url: Optional[str] = None,
    api_token: Optional[str] = None
) -> dict:
    """
    Upload an experiment results archive to the server.
    
    Uploads a pre-compressed archive file to the server's /results/upload endpoint.
    
    Args:
        hash_id: Identifier tying related results together.
        name: Logical name for this experiment.
        archive_path: Path to the archive file (e.g., .tar.gz or .zip).
        experiment_id: Optional experiment ID to tag this result.
        notes: Optional notes for the upload.
        server_url: Server URL (if None, uses default).
        api_token: API token (if None, uses default).
    
    Returns:
        dict: Server response containing status, id, created_at, and experiment_id.
    
    Raises:
        FileNotFoundError: If archive file doesn't exist.
        requests.HTTPError: If upload fails.
    """
    server_url, api_token = _get_defaults(server_url, api_token)
    
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    
    # Read the archive file
    with open(archive_path, 'rb') as f:
        archive_data = f.read()
    
    # Prepare multipart upload
    url = server_url + "/results/upload"
    multipart = {
        "hashID": (None, hash_id),
        "name": (None, name),
        "notes": (None, notes or ""),
        "archive": (os.path.basename(archive_path), archive_data, "application/gzip"),
    }
    
    if experiment_id is not None:
        multipart["experimentID"] = (None, experiment_id)
    
    logger.debug(f"Uploading experiment '{name}' to {url}")
    resp = requests.post(
        url,
        files=multipart,
        headers=_auth_headers(api_token),
        timeout=300
    )
    
    if resp.status_code >= 400:
        raise requests.HTTPError(f"Experiment upload failed ({resp.status_code}): {resp.text}")
    
    return resp.json()


def get_experiments_list(
    hash_id: str,
    server_url: Optional[str] = None,
    api_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all experiment results that share the same hash_id.
    
    Args:
        hash_id: The hash ID to filter by.
        server_url: Server URL (if None, uses default).
        api_token: API token (if None, uses default).
    
    Returns:
        List of dicts with keys: name, experiment_id, notes, created_at.
    
    Raises:
        requests.HTTPError: If the request fails.
    """
    server_url, api_token = _get_defaults(server_url, api_token)
    url = server_url + "/results/list"
    
    resp = requests.get(
        url,
        params={"hashID": hash_id},
        headers=_auth_headers(api_token),
        timeout=120
    )
    
    if resp.status_code >= 400:
        raise requests.HTTPError(f"Results list failed ({resp.status_code}): {resp.text}")
    
    return resp.json().get("items", [])

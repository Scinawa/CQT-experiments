import requests
import argparse
import os
import tarfile
import tempfile
from git.repo.base import Repo


def upload_runcard(dir_path, name, url="http://127.0.0.1:8081/api/upload_runcard"):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    try:
        with tarfile.open(tmp_file.name, "w:gz") as tar:
            tar.add(dir_path, arcname=os.path.basename(dir_path))
        tmp_file.close()

        with open(tmp_file.name, "rb") as f:
            files = {
                "runcard": (
                    f"{os.path.basename(dir_path)}.tar.gz",
                    f,
                    "application/gzip",
                )
            }
            data = {"name": name}
            response = requests.post(url, files=files, data=data)
            if response.ok:
                print(response.json())
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
    finally:
        os.remove(tmp_file.name)



def upload_experiment(
    dir_path,
    runcard_id,
    url="http://127.0.0.1:8081/api/upload_experiment",
):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    experiment_name = os.path.basename(dir_path)
    try:
        with tarfile.open(tmp_file.name, "w:gz") as tar:
            tar.add(dir_path, arcname=os.path.basename(dir_path))
        tmp_file.close()

        with open(tmp_file.name, "rb") as f:
            files = {
                "data": (
                    f"{experiment_name}.tar.gz",
                    f,
                    "application/gzip",
                )
            }
            data = {
                "experiment_name": experiment_name,
                "id": runcard_id,
            }
            response = requests.post(url, files=files, data=data)
            if response.ok:
                print(f"Successfully uploaded experiment: {experiment_name}")
                print(response.json())
            else:
                print(f"Error uploading experiment: {experiment_name}")
                print(f"Error: {response.status_code}")
                print(response.text)
    finally:
        os.remove(tmp_file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload an experiment to the DBReporting API."
    )
    parser.add_argument(
        "directories",
        type=str,
        nargs="+",
        help="The paths to the experiment directories.",
    )

    # parser.add_argument(
    #     "--runcard_id", type=int, help="The ID of the associated runcard.", default=1
    # )

    args = parser.parse_args()



    # obtain current RUNID
    repo = Repo("/mnt/scratch/qibolab_platforms_nqch")
    hash_id = repo.commit().hexsha



    for directory in args.directories:
        upload_experiment(directory, hash_id)

import os
import pathlib

#os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()

from qibo import Circuit, gates, set_backend

import argparse
import json
import sys
from pathlib import Path as _P
import time

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def main(nqubits, device, nshots, **kwargs):

    results = dict()
    data = dict()

    results["readout_fidelity"] = {}
    data["qubits"] = [i for i in range(nqubits)]
    data["nshots"] = nshots
    data["device"] = device

    _tmp_runtimes = []

    set_backend("qibolab", platform=device)

    c1 = Circuit(1)
    c1.add(gates.M(0))

    c2 = Circuit(1)
    c2.add(gates.X(0))
    c2.add(gates.M(0))

    e1 = {}
    e2 = {}

    start_time = time.time()
    for qb in range(20):
        c1._wire_names = [qb]
        c2._wire_names = [qb]

        e1[qb] = float(c1(nshots=nshots).probabilities()[1])
        e2[qb] = float(c2(nshots=nshots).probabilities()[0])
    end_time = time.time()
    _tmp_runtimes.append(end_time - start_time)

    for (qb, gnd_err), exc_err in zip(e1.items(), e2.values()):
        results["readout_fidelity"][f"{qb}"] = 1 - (gnd_err + exc_err) / 2

    runtime_seconds = sum(_tmp_runtimes) / len(_tmp_runtimes) if _tmp_runtimes else 0.0
    results['runtime']= f"{runtime_seconds:.2f} seconds."
    results['description']= f"Readout fidelity experiment executed on {device} backend with {nshots} shots per circuit."

    # Write to data/<scriptname>/<device>/results.json
    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (out_dir / "data.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with (out_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to write output files to {out_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits",
        default=20,
        type=int,
        help="Total number of qubits",
    )
    parser.add_argument(
        "--device",
        choices=["sinq20"],
        default="sinq20",
        type=str,
        help="Device to use",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    parser.add_argument(
        "--qubit_id",
        type=str,
        help="Dummy parameter for compliance",
    )

    args = vars(parser.parse_args())
    main(**args)


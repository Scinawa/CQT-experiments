import numpy as np
import qibo
import os
import json
import sys
from pathlib import Path as _P


sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py



def QFT(qubits_list, device, nshots):
    n_qubits = len(qubits_list)
    total_qubits = int(np.max(qubits_list) + 1)

    circuit = qibo.Circuit(total_qubits)

    # Add Hadamard at the beginning
    for q in qubits_list:
        circuit.add(qibo.gates.H(q))
    # QFT
    qft_circuit = qibo.models.QFT(n_qubits, with_swaps=False)
    circuit.add(qft_circuit.on_qubits(*qubits_list))

    # Add measurement
    for q in qubits_list:
        circuit.add(qibo.gates.M(q))

    result = circuit(nshots=nshots)
    return result


def main(qubits_list, device, nshots):
    # Remove all qibo_client usage and via_client logic
    # Set backend as in template/main.py, GHZ/main.py, mermin/main.py
    if device == "numpy":
        qibo.set_backend("numpy")
    else:
        qibo.set_backend("qibolab", platform=device)

    results = dict()
    data = dict()

    results["success_rate"] = {}
    results["plotparameters"] = {}
    results["plotparameters"]["frequencies"] = {}
    data["qubits_list"] = qubits_list
    data["nshots"] = nshots
    data["device"] = device

    result = QFT(qubits_list, device, nshots)

    n_qubits = len(qubits_list)
    success_keys = ["0" * n_qubits, "1" * n_qubits]
    total_success = sum(result.frequencies().get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    all_bitstrings = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
    freq_dict = {bitstr: result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings}

    results = {
        "success_rate": success_rate,
        "plotparameters": {"frequencies": freq_dict},
    }

    out_dir = config.output_dir_for(__file__) / device
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(out_dir, f"results.json")

    with open(output_path, "w") as f:
        json.dump(results, f)
        print(f'File saved on {output_path}')

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubits_list",
        default=[0, 1, 5],
        type=int,
        nargs='+',
        help="List of qubits exploited in the device",
    )
    parser.add_argument(
        "--device",
        default="numpy",
        type=str,
        help="Device to use (e.g., 'nqch' or 'numpy' for local simulation)",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    args = vars(parser.parse_args())
    main(args["qubits_list"], args["device"], args["nshots"])
import json
import numpy as np
from qibo import Circuit, gates, set_backend
import os
import argparse
import sys
from pathlib import Path as _P
import time
import ast

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

from qibocal.auto.execute import Executor
from qibocal.cli.report import report


def create_ghz_circuit(qubits_list):
    nqubits = len(qubits_list)
    c = Circuit(nqubits)
    c.add(gates.H(qubits_list[0]))
    for i in range(nqubits - 1):
        c.add(gates.CNOT(qubits_list[i], qubits_list[i + 1]))
    for qubit in qubits_list:
        c.add(gates.M(qubit))
    return c


def prepare_ghz_results(frequencies, nshots, nqubits):
    # Calculate success rate for GHZ state (all 0s or all 1s)
    success_keys = ["0" * nqubits, "1" * nqubits]
    total_success = sum(frequencies.get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    # Prepare output structure
    all_bitstrings = [format(i, f"0{nqubits}b") for i in range(2**nqubits)]
    freq_dict = {bitstr: frequencies.get(bitstr, 0) for bitstr in all_bitstrings}
    return {"success_rate": success_rate, "plotparameters": {"frequencies": freq_dict}}


def run_ghz_experiment(qubits_list, device, nshots, root_path):
    nqubits = len(qubits_list)

    if device == "numpy":
        set_backend("numpy")
        platform = None
    else:
        set_backend("qibolab", platform=device)
        platform = device

    results = {}
    data = {
        "qubits_list": qubits_list,
        "nshots": nshots,
        "device": device,
    }

    # Only run Executor if using qibolab
    if platform:
        with Executor.open(
            "myexec",
            path=root_path,
            platform=platform,
            targets=qubits_list,
            update=True,
            force=True,
        ) as e:
            circuit = create_ghz_circuit(qubits_list)
            start_time = time.time()
            # For GHZ, we'll run the circuit directly since qibocal doesn't have a specific GHZ method
            # You might need to adapt this based on available qibocal methods
            result = circuit(nshots=nshots)
            end_time = time.time()
            runtime_seconds = end_time - start_time
            report(e.path, e.history)

            frequencies = result.frequencies()
            ghz_results = prepare_ghz_results(frequencies, nshots, nqubits)

            results.update(ghz_results)
            results["runtime"] = f"{runtime_seconds:.2f} seconds."
            results["qubits_used"] = qubits_list
            results["description"] = (
                f"GHZ circuit with {nqubits} qubits executed on {device} backend with {nshots} shots."
            )
    else:
        # Simulate with numpy backend
        start_time = time.time()
        circuit = create_ghz_circuit(qubits_list)
        result = circuit(nshots=nshots)
        end_time = time.time()
        runtime_seconds = end_time - start_time

        frequencies = result.frequencies()
        ghz_results = prepare_ghz_results(frequencies, nshots, nqubits)

        results.update(ghz_results)
        results["description"] = (
            f"GHZ circuit with {nqubits} qubits executed on numpy backend with {nshots} shots."
        )
        results["runtime"] = f"{runtime_seconds:.2f} seconds."
        results["qubits_used"] = qubits_list

    return data, results


def main(device, nshots, qubits_list):
    scriptname = _P(__file__).stem
    out_dir = config.output_dir_for(__file__, device)
    out_dir_tmp = out_dir / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        data, results = run_ghz_experiment(qubits_list, device, nshots, out_dir_tmp)
    except Exception as e:
        print(f"Failed to run GHZ experiment on qubits {qubits_list}: {e}")
        return

    try:
        with (out_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to write output files to {out_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubits_list",
        type=str,
        default="[0, 1, 2]",
        help="List of qubits to use for GHZ circuit as string representation",
    )
    parser.add_argument("--nshots", type=int, default=1000)
    parser.add_argument("--device", choices=["numpy", "sinq20"], default="numpy")
    args = parser.parse_args()

    # Parse the qubit list string into actual list of integers
    try:
        qubits_list = ast.literal_eval(args.qubits_list)
        # Ensure all elements are integers
        qubits_list = [int(q) for q in qubits_list]
    except (ValueError, SyntaxError, TypeError):
        print(f"Error: Invalid qubit list format: {args.qubits_list}")
        sys.exit(1)

    main(args.device, args.nshots, qubits_list)

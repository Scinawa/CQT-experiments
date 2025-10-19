import argparse
from qibo import Circuit, gates, set_backend
import json
from pathlib import Path
import sys
from pathlib import Path as _P
import time
import ast

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def grover_2q(qubits, target):
    c = Circuit(20)
    c.add([gates.H(i) for i in qubits])
    for i, bit in enumerate(target):
        if int(bit) == 0:
            c.add(gates.X(qubits[i]))
    c.add(gates.CZ(qubits[0], qubits[1]))
    for i, bit in enumerate(target):
        if int(bit) == 0:
            c.add(gates.X(qubits[i]))
    c.add([gates.H(i) for i in qubits])
    c.add([gates.X(i) for i in qubits])
    c.add(gates.CZ(qubits[0], qubits[1]))
    c.add([gates.X(i) for i in qubits])
    c.add([gates.H(i) for i in qubits])
    c.add(gates.M(*qubits, register_name=f"m{qubits}"))
    return c


def main(qubit_edge_list, device, nshots):
    if device == "numpy":
        set_backend("numpy")
    else:
        set_backend("qibolab", platform=device)

    results = dict()
    data = dict()

    target = "11"

    qubits_list = qubit_edge_list[0]

    results["success_rate"] = {}
    results["plotparameters"] = {}
    results["plotparameters"]["frequencies"] = {}
    data["qubit_edge_list"] = qubit_edge_list
    data["nshots"] = nshots
    data["device"] = device
    data["target"] = target

    _tmp_runtimes = []

    c = grover_2q(qubits_list, target)

    start_time = time.time()
    r = c(nshots=nshots)
    end_time = time.time()
    _tmp_runtimes.append(end_time - start_time)

    freq = r.frequencies()

    target_freq = freq.get(target, 0)
    results["success_rate"][f"{qubits_list}"] = target_freq / nshots

    # Make probabilities a dict keyed by all possible bitstrings
    num_bits = len(qubits_list)
    all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
    prob_dict = {bs: (freq.get(bs, 0) / nshots) for bs in all_bitstrings}
    results["plotparameters"]["frequencies"][f"{qubits_list}"] = prob_dict

    runtime_seconds = sum(_tmp_runtimes) / len(_tmp_runtimes) if _tmp_runtimes else 0.0
    results["runtime"] = f"{runtime_seconds:.2f} seconds."
    results["description"] = (
        f"Grover's algorithm for 2 qubits executed on {device} backend with {nshots} shots per circuit. \n We measure the success rate of finding the target state '{target}' for each pair of qubits in {qubits_list}."
    )
    results["qubits_used"] = qubits_list

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
        "--qubit_edge_list",
        default="[[13, 14]]",
        type=str,
        help="Target qubit list as string representation",
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "sinq20"],
        default="numpy",
        type=str,
        help="Device to use (numpy or sinq20)",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    args = parser.parse_args()

    # Parse the qubit list string into actual list of integers
    try:
        qubits_list = ast.literal_eval(args.qubits_list)
        # Ensure all elements are integers
        qubits_list = [int(q) for q in qubits_list]
    except (ValueError, SyntaxError, TypeError):
        print(f"Error: Invalid qubit list format: {args.qubits_list}")
        sys.exit(1)

    main(qubits_list, args.device, args.nshots)

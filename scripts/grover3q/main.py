import argparse
from qibo import Circuit, gates, set_backend
import json
from pathlib import Path
import sys
import time
from pathlib import Path as _P
import ast

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def ccz_gate_auxilliary():
    """ Connectivity is the following [[0, 2], [2, 1]]
    """
    ccz = Circuit(3)

    ccz.add(gates.CNOT(1, 2))
    ccz.add(gates.TDG(2))
    ccz.add(gates.CNOT(0, 2))
    ccz.add(gates.T(2))
    ccz.add(gates.CNOT(1, 2))
    ccz.add(gates.TDG(2))
    ccz.add(gates.CNOT(0, 2))
    ccz.add(gates.T(2))

    ccz.add(gates.CNOT(0, 2))
    ccz.add(gates.CNOT(2, 0))
    ccz.add(gates.CNOT(0, 2))

    ccz.add(gates.T(1))
    ccz.add(gates.CNOT(2, 1))
    ccz.add(gates.TDG(1))
    ccz.add(gates.T(2))
    ccz.add(gates.CNOT(2, 1))

    ccz.add(gates.CNOT(0, 2))
    ccz.add(gates.CNOT(2, 0))
    ccz.add(gates.CNOT(0, 2))

    return ccz


def grover_3q(qubits, target):
    ccz = ccz_gate_auxilliary()

    qubits = qubits

    c = Circuit(20)
    c.add([gates.H(i) for i in qubits])
    for i, bit in enumerate(target):
        if int(bit) == 0:
            c.add(gates.X(qubits[i]))
    c.add(ccz.on_qubits(*qubits))
    for i, bit in enumerate(target):
        if int(bit) == 0:
            c.add(gates.X(qubits[i]))
    c.add([gates.H(i) for i in qubits])
    c.add([gates.X(i) for i in qubits])
    c.add(ccz.on_qubits(*qubits))
    c.add([gates.X(i) for i in qubits])
    c.add([gates.H(i) for i in qubits])

    c.add(gates.M(*qubits, register_name=f"m{qubits}"))
    # print(c.draw())  # Optional: comment out or remove for production
    return c


def build_qubits_from_edges(qubit_edge_list):
    """ Given 3 qubits in this connection a-b-c, generate qubit list in the order 0-2-1.
    """
    assert len(list(set(qubit_edge_list[0])&set(qubit_edge_list[1])))==1
    assert len(list(set(qubit_edge_list[0])^set(qubit_edge_list[1])))==2
    assert len(qubit_edge_list)==2
    
    qubits = list(set(qubit_edge_list[0])^set(qubit_edge_list[1]))
    qubits += list(set(qubit_edge_list[0])&set(qubit_edge_list[1]))
    return qubits


def main(qubit_edge_list, device, nshots):
    if device == "numpy":
        set_backend("numpy")
    else:
        set_backend("qibolab", platform=device)

    results = dict()
    data = dict()

    target = "111"

    results["success_rate"] = {}
    results["plotparameters"] = {}
    results["plotparameters"]["frequencies"] = {}
    data["qubit_pairs"] = qubit_edge_list
    data["nshots"] = nshots
    data["device"] = device
    data["target"] = target

    _tmp_runtimes = []

    qubits = build_qubits_from_edges(qubit_edge_list)

    c = grover_3q(qubits, target)

    start_time = time.time()
    r = c(nshots=nshots)
    end_time = time.time()
    _tmp_runtimes.append(end_time - start_time)

    freq = r.frequencies()

    target_freq = freq.get(target, 0)
    results["success_rate"][f"{qubits}"] = target_freq / nshots

    # Make probabilities a dict keyed by all possible bitstrings
    num_bits = len(qubits) - 1  # Only measure main qubits, not ancilla
    all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
    prob_dict = {bs: (freq.get(bs, 0) / nshots) for bs in all_bitstrings}
    results["plotparameters"]["frequencies"][f"{qubits}"] = prob_dict

    runtime_seconds = sum(_tmp_runtimes) / len(_tmp_runtimes) if _tmp_runtimes else 0.0
    results["runtime"] = f"{runtime_seconds:.2f} seconds."
    results["description"] = (
        f"Grover's algorithm for 3 qubits executed on {device} backend with {nshots} shots per circuit. \n We measure the success rate of finding the target state '{target}' for each pair of qubits in {qubit_groups}."
    )
    results["qubits_used"] = qubit_groups

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
        default="[[18, 14], [18, 19]]",
        type=str,
        help="Target qubits as string representation, last qubit used as ancilla",
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "sinq20"],
        default="numpy",
        type=str,
        help="Device to use (e.g., 'sinq20' or 'numpy' for local simulation)",
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
    
    main([qubits_list], args.device, args.nshots)

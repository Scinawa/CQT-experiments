import numpy as np
import qibo
import os
import json
import sys
import time
from pathlib import Path as _P

''' Quantum Fourier Transform (QFT) implementation 
    using qibo framework with !AUTOMATIC TRANSPILATION!
'''


sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def QFT(qubits_list):
    '''
    Quantum Fourier Transform circuit with AUTO TRANSPILATION (ONLY THREE QUBITS)
    
    Args:   - list of edges (connections)
            
    Returns: qibo circuit
    '''
    
    qubits_list = sorted(set(sum(qubits_list, [])))
    total_qubits = int(max(qubits_list) + 1)     # number of total qubits of the circuit

    n_qubits = len(qubits_list)

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

    return circuit

def main(qubits_list, device, nshots):
    # Remove all qibo_client usage and via_client logic
    # Set backend as in template/main.py, GHZ/main.py, mermin/main.py
    if device == "numpy":
        qibo.set_backend("numpy")
    else:
        qibo.set_backend("qibolab", platform=device)

    # Here the list of the best three qubits
    
    qubits_set = set(sum(qubits_list, []))
    num_qubits = len(qubits_set)                         # number of qubits 

    frequencies = dict()
    all_bitstrings = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]

    print(f"Trying edges: {qubits_list}")

    circuit = QFT(qubits_list)

    start = time.perf_counter()
    result = circuit(nshots=nshots)
    end = time.perf_counter()

    # Circuit output frequencies and store them
    frequencies = {bitstr: result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings}

    num_gates = len(circuit.queue)
    depth = circuit.depth

    results = dict()
    data = dict()

    results["description"] = {}
    results["circuit_depth"] = {}
    results["gates_count"] = {}
    results["duration"] = {}
    results["frequencies"] = {}
    results["edges"] = {}
    data["nshots"] = nshots
    data["device"] = device

    results = {
        "description": f"Implementation of the Quantum Fourier Transform on three qubits with automatic transpilation. The number of gates is {num_gates}, the depth of the circuit is {depth}",
        "circuit_depths": depth,
        "gates_counts": num_gates,
        "duration": f"{(end-start):.3f} seconds.",
        "frequencies": frequencies,
        "edges": qubits_list,
    }
    
    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(out_dir, f"results.json")

    with open(output_path, "w") as f:
        json.dump(results, f)
        print(f"File saved on {output_path}")

import argparse
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--qubits_list",
        default="[[9,8],[8,13]]",
        type=str,
        help="Target edges list as string representation",
    )
    args = parser.parse_args()
    # Parse the qubit list string into actual list of integers
    try:
        qubits_list = ast.literal_eval(args.qubits_list)
        # Ensure all elements are integers
        qubits_list = [[int(q) for q in edge] for edge in qubits_list]
    except (ValueError, SyntaxError, TypeError):
        print(f"Error: Invalid qubit list format: {args.qubits_list}")
        sys.exit(1)
    main(qubits_list, args.device, args.nshots)
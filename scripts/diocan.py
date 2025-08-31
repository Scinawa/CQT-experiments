import numpy as np
import qibo
from qibo import gates
import os
import json
import sys
from pathlib import Path as _P
import argparse
import pickle
from collections import Counter
from itertools import islice
from qibo.ui import plot_circuit
import matplotlib.pyplot as plt
import logging

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

logger = logging.getLogger(__name__)


# def var_circuit(angles_dict, num_qubits=3, num_layers=1):
#     """
#     Build a quantum circuit based on the angles dictionary from the JSON file.

#     Interpretation of depths (matches original PennyLane layout / provided JSON):
#       - depth '1' : initial RY on the qubit
#       - depth '2' : initial RZ on the qubit
#       - depth '3' : layer RY on the qubit
#       - depth '4' : entangling RZ applied to the *target* qubit after CNOT(i,i+1)
#     angles_dict uses qubit labels starting at "1" (map to wire index q = int(key)-1).
#     """
#     circuit = qibo.Circuit(num_qubits)

#     # initial rotations (depths 1 and 2)
#     for q in range(num_qubits):
#         qkey = str(q + 1)
#         q_angles = angles_dict.get(qkey, {})
#         print(q_angles)
#         if "1" in q_angles:
#             angle = q_angles["1"]
#             print(f"Adding RY (depth 1) on qubit {q} with angle {angle}")
#             circuit.add(gates.RY(q, theta=angle))
#         if "2" in q_angles:
#             angle = q_angles["2"]
#             print(f"Adding RZ (depth 2) on qubit {q} with angle {angle}")
#             circuit.add(gates.RZ(q, theta=angle))
#         if "3" in q_angles:
#             angle = q_angles["3"]
#             print(f"Adding RY (depth 3) on qubit {q} with angle {angle}")
#             circuit.add(gates.RY(q, theta=angle))

#     circuit.add(gates.CNOT(0, 1))
#     circuit.add(gates.RZ(1, theta=angles_dict["2"]["4"]))
#     circuit.add(gates.CNOT(0, 1))

#     circuit.add(gates.CNOT(1, 2))
#     circuit.add(gates.RZ(2, theta=angles_dict["3"]["4"]))
#     circuit.add(gates.CNOT(1, 2))

#     circuit.add(gates.RY(0, theta=angles_dict["1"]["4"]))
#     circuit.add(gates.RY(1, theta=angles_dict["2"]["5"]))
#     circuit.add(gates.RY(2, theta=angles_dict["3"]["5"]))

#     circuit.add(gates.CNOT(0, 1))
#     circuit.add(gates.RZ(1, theta=angles_dict["2"]["6"]))
#     circuit.add(gates.CNOT(0, 1))

#     circuit.add(gates.CNOT(1, 2))
#     circuit.add(gates.RZ(2, theta=angles_dict["3"]["6"]))
#     circuit.add(gates.CNOT(1, 2))

#     #import pdb; pdb.set_trace()
    
#     return circuit


def var_circuit(angles_dict, num_qubits, num_layers=None):
    """
    Build a quantum circuit based on the angles dictionary from the JSON file.
    This follows the exact same structure as the PennyLane reference implementation.
    
    angles_dict uses qubit labels "0", "1", "2" (zero-indexed strings).
    """
    circuit = qibo.Circuit(num_qubits)

    if num_qubits==3:
        for q in range(num_qubits):
            qkey = str(q)
            q_angles = angles_dict.get(qkey, {})
            
            logger.debug("Adding RY (depth 1) on qubit %d with angle %s", q, q_angles["1"])
            circuit.add(gates.RY(q, theta=q_angles["1"]))
        
            logger.debug("Adding RZ (depth 2) on qubit %d with angle %s", q, q_angles["2"])
            circuit.add(gates.RZ(q, theta=q_angles["2"]))
            
            logger.debug("Adding RY (depth 3) on qubit %d with angle %s", q, q_angles["3"])
            circuit.add(gates.RY(q, theta=q_angles["3"]))

        # First entangling layer
        # CNOT 0->1; RZ on qubit 1 using angles_dict["1"]["4"]; CNOT
        circuit.add(gates.CNOT(0, 1))
        logger.debug("Adding RZ (depth 4) on qubit 1 with angle %s", angles_dict["1"]["4"])
        circuit.add(gates.RZ(1, theta=angles_dict["1"]["4"]))
        circuit.add(gates.CNOT(0, 1))

        # CNOT 1->2; RZ on qubit 2 using angles_dict["2"]["4"]; CNOT  
        circuit.add(gates.CNOT(1, 2))
        logger.debug("Adding RZ (depth 4) on qubit 2 with angle %s", angles_dict["2"]["4"])
        circuit.add(gates.RZ(2, theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(1, 2))

        # RY rotations using depths 4 and 5 (note: qubit 0 uses depth 4, others use depth 5)
        logger.debug("Adding RY (depth 4) on qubit 0 with angle %s", angles_dict["0"]["4"])
        circuit.add(gates.RY(0, theta=angles_dict["0"]["4"]))

        logger.debug("Adding RY (depth 5) on qubit 1 with angle %s", angles_dict["1"]["5"])
        circuit.add(gates.RY(1, theta=angles_dict["1"]["5"]))

        logger.debug("Adding RY (depth 5) on qubit 2 with angle %s", angles_dict["2"]["5"])
        circuit.add(gates.RY(2, theta=angles_dict["2"]["5"]))

        # Second entangling layer  
        # CNOT 0->1; RZ on qubit 1 using angles_dict["1"]["6"]; CNOT
        circuit.add(gates.CNOT(0, 1))
        circuit.add(gates.RZ(1, theta=angles_dict["1"]["6"]))
        circuit.add(gates.CNOT(0, 1))

        # CNOT 1->2; RZ on qubit 2 using angles_dict["2"]["6"]; CNOT
        circuit.add(gates.CNOT(1, 2))
        circuit.add(gates.RZ(2, theta=angles_dict["2"]["6"]))
        circuit.add(gates.CNOT(1, 2))

    if num_qubits==4:
        for q in range(num_qubits):
            qkey = str(q)
            q_angles = angles_dict.get(qkey, {})
            
            logger.debug("Adding RY (depth 1) on qubit %d with angle %s", q, q_angles["1"])
            circuit.add(gates.RY(q, theta=q_angles["1"]))
        
            logger.debug("Adding RZ (depth 2) on qubit %d with angle %s", q, q_angles["2"])
            circuit.add(gates.RZ(q, theta=q_angles["2"]))
            
            logger.debug("Adding RY (depth 3) on qubit %d with angle %s", q, q_angles["3"])
            circuit.add(gates.RY(q, theta=q_angles["3"]))

        # First entangling layer
        # CNOT 0->1; RZ on qubit 1 using angles_dict["1"]["4"]; CNOT
        circuit.add(gates.CNOT(0, 1))
        logger.debug("Adding RZ (depth 4) on qubit 1 with angle %s", angles_dict["1"]["4"])
        circuit.add(gates.RZ(1, theta=angles_dict["1"]["4"]))
        circuit.add(gates.CNOT(0, 1))

        # CNOT 1->2; RZ on qubit 2 using angles_dict["2"]["4"]; CNOT  
        circuit.add(gates.CNOT(1, 2))
        logger.debug("Adding RZ (depth 4) on qubit 2 with angle %s", angles_dict["2"]["4"])
        circuit.add(gates.RZ(2, theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(1, 2))

        # CNOT 1->2; RZ on qubit 2 using angles_dict["2"]["4"]; CNOT  
        circuit.add(gates.CNOT(2, 3))
        logger.debug("Adding RZ (depth 4) on qubit 2 with angle %s", angles_dict["2"]["4"])
        circuit.add(gates.RZ(3, theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(2, 3))


    # measurements are added later by main (we avoid measure_all here)
    return circuit

def load_data(config_file):
    """
    Load the configuration data from the JSON file.
    """
    with open(config_file, 'r') as f:
        data = json.load(f)
    return data

def main(qubits_list, device, nshots, debug=False, args=None):
    # configure logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Set backend
    if device == "numpy":
        qibo.set_backend("numpy")
        logger.info("Using backend: numpy")
    else:
        qibo.set_backend("qibolab", platform=device)
        logger.info("Using backend: qibolab (platform=%s)", device)
    
    config_path = "scripts/qml-yeast_class_3q/exp_1_3Q_2L_yeast_best.json"
    logger.info("Loading experiment configuration from %s", config_path)
    data = load_data(config_path)

    
    # Extract circuit parameters
    args = data['args']
    num_qubits = args['num_qubits']
    num_layers = args['num_layers']
    output_qubit = args['output_qubit']

    results = []
    
    # Process each data point in the dataset
    for idx, config_data in islice(data['qc_configurations'].items(), None):
        angles_dict = config_data['angles']
        true_label = config_data['label']
        noiseless_label = config_data['noiseless_label']
        
        # Create circuit for this data point
        logger.debug("Building circuit for data index %s; true=%s noiseless=%s", idx, true_label, noiseless_label)
        #logger.debug("Angles dict (truncated): %s", {k: angles_dict.get(k) for k in list(angles_dict)[:4]})
        circuit = var_circuit(angles_dict, num_qubits, num_layers)
        
        # Add measurement to the output qubit only and run
        circuit.add(gates.M(output_qubit))

        result = circuit(nshots=nshots)

        frequencies = result.frequencies()
        total_counts = sum(frequencies.values())
        if total_counts == 0:
            logger.warning("No counts returned for idx %s (total_counts=0)", idx)
         
        c1 = frequencies.get('1', 0)
        c0 = frequencies.get('0', 0)
        total_counts = c0 + c1
        if total_counts > 0:
            exp_z = (c0 - c1) / total_counts  # <Z> = P(0) - P(1)
        else:
            exp_z = 0.0
        one_prob = 1.0 / (1.0 + np.exp(-exp_z))  # sigmoid(<Z>)
        predicted_label = 1 if one_prob > 0.5 else 0


        # Store the results
        result_entry = {
            "data_point_idx": idx,
            "true_label": true_label,
            "noiseless_label": noiseless_label,
            "predicted_label": predicted_label,
            "prediction_probability": one_prob,
            "correct": predicted_label == true_label
        }
        results.append(result_entry)
        
        logger.info("Processed data point %s: predicted=%s true=%s noiseless=%s prob=%.3f",
                    idx, predicted_label, true_label, noiseless_label, one_prob)
 
    # Calculate overall accuracy


    accuracy = sum(r["correct"] for r in results) / len(results)
    logger.info("Our accuracy: %.4f", accuracy)
    
    # Calculate noiseless accuracy (compare predicted with noiseless)
    noiseless_accuracy = sum(r["predicted_label"] == r["noiseless_label"] for r in results) / len(results)
    logger.info("Accuracy in being equal to noiseless (should be 1): %.4f", noiseless_accuracy)

    original_sim_accuracy = sum(r["true_label"] == r["noiseless_label"] for r in results) / len(results)
    logger.info("Accuracy in being equal to original noiseless simulation: %.4f", original_sim_accuracy)
    logger.info("Should be: %.4f", data['accuracy'])


    # dump the results in a file
    out_dir = config.output_dir_for(__file__) / device
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(out_dir, f"results.json")

    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "noiseless_accuracy": noiseless_accuracy,
            "details": results
        }, f, indent=2)
        logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubits_list",
        default=[0, 1, 2],
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
        default=300,
        type=int,
        help="Number of shots for each circuit execution",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output",
    )
    cnf = vars(parser.parse_args())
    main(cnf["qubits_list"], cnf["device"], cnf["nshots"], cnf.get("debug", False))
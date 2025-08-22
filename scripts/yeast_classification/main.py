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

def var_circuit(angles_dict, num_qubits=4, num_layers=1):
    """
    Build a quantum circuit based on the angles dictionary from the JSON file.

    Interpretation of depths (matches original PennyLane layout / provided JSON):
      - depth '1' : initial RY on the qubit
      - depth '2' : initial RZ on the qubit
      - depth '3' : layer RY on the qubit
      - depth '4' : entangling RZ applied to the *target* qubit after CNOT(i,i+1)
    angles_dict uses qubit labels starting at "1" (map to wire index q = int(key)-1).
    """
    circuit = qibo.Circuit(num_qubits)

    # initial rotations (depths 1 and 2)
    for q in range(num_qubits):
        qkey = str(q + 1)
        q_angles = angles_dict.get(qkey, {})
        if "1" in q_angles:
            angle = q_angles["1"]
            logger.debug("Adding RY (depth 1) on qubit %d with angle %s", q, angle)
            circuit.add(gates.RY(q, theta=angle))
        if "2" in q_angles:
            angle = q_angles["2"]
            logger.debug("Adding RZ (depth 2) on qubit %d with angle %s", q, angle)
            circuit.add(gates.RZ(q, theta=angle))

    # layered RY (depth 3)
    for l in range(num_layers):
        # depth index for layer RY in the provided JSON is '3' (single layer case)
        for q in range(num_qubits):
            qkey = str(q + 1)
            q_angles = angles_dict.get(qkey, {})
            if "3" in q_angles:
                angle = q_angles["3"]
                logger.debug("Adding RY (depth 3, layer %d) on qubit %d with angle %s", l, q, angle)
                circuit.add(gates.RY(q, theta=angle))

        # entangling sequence with conditional RZ on the target (depth 4)
        for i in range(num_qubits - 1):
            circuit.add(gates.CNOT(i, i + 1))
            tgt_key = str(i + 2)  # target qubit key is i+1 in zero-based -> +2 as string
            tgt_angles = angles_dict.get(tgt_key, {})
            if "4" in tgt_angles:
                angle = tgt_angles["4"]
                logger.debug("Adding RZ (depth 4) on target qubit %d with angle %s", i + 1, angle)
                circuit.add(gates.RZ(i + 1, theta=angle))
            circuit.add(gates.CNOT(i, i + 1))


    # measurements are added later by main (we avoid measure_all here)
    return circuit

def load_data(config_file):
    """
    Load the configuration data from the JSON file.
    """
    with open(config_file, 'r') as f:
        data = json.load(f)
    return data

def main(qubits_list, device, nshots, debug=False):
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
    
    # Load configuration data
    config_path =  "scripts/yeast_classification/exp_1_4Q_1L_statlog_best.pkl_pretty"
    logger.info("Loading experiment configuration from %s", config_path)
    data = load_data(config_path)
    
    # Extract circuit parameters
    args = data['args']
    num_qubits = args.get('num_qubits', 0)  # Default to 3 if not specified
    num_layers = args.get('num_layers', 0)   # Default to 1 if not specified
    output_qubit = args.get('output_qubit', 0)  # Default qubit to measure
    #print("Output qubit:", output_qubit)
    #import pdb
    #pdb.set_trace()

    # Store results
    results = []
    
    # Process each data point in the dataset
    for idx, config_data in islice(data['qc_configurations'].items(), None):
        angles_dict = config_data['angles']
        true_label = config_data['label']
        noiseless_label = config_data['noiseless_label']
        
        # Create circuit for this data point
        logger.debug("Building circuit for data index %s; true=%s noiseless=%s", idx, true_label, noiseless_label)
        logger.debug("Angles dict (truncated): %s", {k: angles_dict.get(k) for k in list(angles_dict)[:4]})
        circuit = var_circuit(angles_dict, num_qubits, num_layers)
        
        # Add measurement to the output qubit only and run
        circuit.add(gates.M(output_qubit))
        logger.debug("Executing circuit (nshots=%d) for data index %s", nshots, idx)
        result = circuit(nshots=nshots)
        frequencies = result.frequencies()
        total_counts = sum(frequencies.values())
        if total_counts == 0:
            logger.warning("No counts returned for idx %s (total_counts=0)", idx)
        # Since we're only measuring one qubit, keys are '0'/'1'
        one_prob = frequencies.get('1', 0) / (total_counts if total_counts > 0 else 1)
         
        # Determine predicted label (threshold 0.5)
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
    logger.info("Overall accuracy: %.4f", accuracy)
    
    # Calculate noiseless accuracy (compare predicted with noiseless)
    noiseless_accuracy = sum(r["predicted_label"] == r["noiseless_label"] for r in results) / len(results)
    logger.info("Noiseless comparison accuracy: %.4f", noiseless_accuracy)
    
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
        default=50,
        type=int,
        help="Number of shots for each circuit execution",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output",
    )
    args = vars(parser.parse_args())
    main(args["qubits_list"], args["device"], args["nshots"], args.get("debug", False))
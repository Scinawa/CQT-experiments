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

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

def var_circuit(angles_dict, num_qubits=3, num_layers=1):
    """
    Build a quantum circuit based on the angles dictionary from the JSON file.
    
    Args:
        angles_dict: Dictionary where keys are qubits and values are dictionaries 
                     mapping depth positions to angle values
        num_qubits: Number of qubits in the circuit (default: 3)
        num_layers: Number of layers in the circuit (default: 1)
    """
    circuit = qibo.Circuit(num_qubits)
    
    # Initial RY and RZ rotations
    for i in range(num_qubits):
        # Check if qubit i has angles defined
        if str(i) in angles_dict:
            # Apply RY gates if defined in the angles
            if '0' in angles_dict[str(i)]:
                circuit.add(gates.RY(i, theta=angles_dict[str(i)]['0']))
            # Apply RZ gates if defined in the angles
            if '1' in angles_dict[str(i)]:
                circuit.add(gates.RZ(i, theta=angles_dict[str(i)]['1']))
    
    # Apply layered structure
    for l in range(num_layers):
        layer_offset = 2 + l*2
        
        # Apply RY gates for each qubit in this layer
        for i in range(num_qubits):
            if str(i) in angles_dict and str(layer_offset) in angles_dict[str(i)]:
                circuit.add(gates.RY(i, theta=angles_dict[str(i)][str(layer_offset)]))
        
        # Apply entangling gates and RZ rotations
        for i in range(num_qubits - 1):
            circuit.add(gates.CNOT(i, i + 1))
            if str(i+1) in angles_dict and str(layer_offset+1) in angles_dict[str(i+1)]:
                circuit.add(gates.RZ(i + 1, theta=angles_dict[str(i+1)][str(layer_offset+1)]))
            circuit.add(gates.CNOT(i, i + 1))
    
    # Add measurements
    #circuit.measure_all()
    
    return circuit

def load_data(config_file):
    """
    Load the configuration data from the JSON file.
    """
    with open(config_file, 'r') as f:
        data = json.load(f)
    return data

def main(qubits_list, device, nshots):
    # Set backend
    if device == "numpy":
        qibo.set_backend("numpy")
    else:
        qibo.set_backend("qibolab", platform=device)
    
    # Load configuration data
    config_path =  "scripts/yeast_classification/exp_1_4Q_1L_statlog_best.pkl_pretty"
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
    for idx, config_data in islice(data['qc_configurations'].items(), 20):
        angles_dict = config_data['angles']
        true_label = config_data['label']
        noiseless_label = config_data['noiseless_label']
        
        # Create circuit for this data point
        circuit = var_circuit(angles_dict, num_qubits, num_layers)
        
        # Add measurement to the output qubit only
        circuit.add(gates.M(output_qubit))
        
        # Execute circuit once with the specified number of shots
        result = circuit(nshots=nshots)
        frequencies = result.frequencies()
        
        # Calculate the probability of measuring |1⟩ on the output qubit
        one_prob = 0
        total_counts = sum(frequencies.values())
        
        # Since we're only measuring one qubit, the keys in frequencies 
        # are simply '0' and '1' for that qubit's states
        one_prob = frequencies.get('1', 0)
        one_prob /= total_counts if total_counts > 0 else 1
        
        # Determine predicted label (1 if probability > 0.5, else 0)
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
        
        print(f"Processed data point {idx}: Predicted {predicted_label}, True {true_label}, Noiseless {noiseless_label}")

    # Calculate overall accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Calculate noiseless accuracy (compare predicted with noiseless)
    noiseless_accuracy = sum(r["predicted_label"] == r["noiseless_label"] for r in results) / len(results)
    print(f"Noiseless comparison accuracy: {noiseless_accuracy:.4f}")
    
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
        print(f'Results saved to {output_path}')


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
    args = vars(parser.parse_args())
    main(args["qubits_list"], args["device"], args["nshots"])
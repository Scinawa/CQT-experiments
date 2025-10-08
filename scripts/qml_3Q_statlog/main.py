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
import signal
import shutil
import atexit
import time

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

logger = logging.getLogger(__name__)

def var_circuit(angles_dict, qubits_list):
    """
    Build a quantum circuit based on the angles dictionary from the JSON file.
    Now maps to physical qubits specified in qubits_list.
    """
    num_qubits = len(qubits_list)
    max_qubit = max(qubits_list)
    circuit = qibo.Circuit(max_qubit + 1)

    if num_qubits == 3:
        for i, physical_qubit in enumerate(qubits_list):
            qkey = str(i)
            q_angles = angles_dict.get(qkey, {})
            
            logger.debug("Adding RY (depth 1) on physical qubit %d with angle %s", physical_qubit, q_angles["1"])
            circuit.add(gates.RY(physical_qubit, theta=q_angles["1"]))
        
            logger.debug("Adding RZ (depth 2) on physical qubit %d with angle %s", physical_qubit, q_angles["2"])
            circuit.add(gates.RZ(physical_qubit, theta=q_angles["2"]))
            
            logger.debug("Adding RY (depth 3) on physical qubit %d with angle %s", physical_qubit, q_angles["3"])
            circuit.add(gates.RY(physical_qubit, theta=q_angles["3"]))

        # First entangling layer
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))
        circuit.add(gates.RZ(qubits_list[1], theta=angles_dict["1"]["4"]))
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))

        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))
        circuit.add(gates.RZ(qubits_list[2], theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))

        # RY rotations
        circuit.add(gates.RY(qubits_list[0], theta=angles_dict["0"]["4"]))
        circuit.add(gates.RY(qubits_list[1], theta=angles_dict["1"]["5"]))
        circuit.add(gates.RY(qubits_list[2], theta=angles_dict["2"]["5"]))

        # Second entangling layer  
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))
        circuit.add(gates.RZ(qubits_list[1], theta=angles_dict["1"]["6"]))
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))

        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))
        circuit.add(gates.RZ(qubits_list[2], theta=angles_dict["2"]["6"]))
        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))

    elif num_qubits == 4:
        for i, physical_qubit in enumerate(qubits_list):
            qkey = str(i)
            q_angles = angles_dict.get(qkey, {})
            
            circuit.add(gates.RY(physical_qubit, theta=q_angles["1"]))
            circuit.add(gates.RZ(physical_qubit, theta=q_angles["2"]))
            circuit.add(gates.RY(physical_qubit, theta=q_angles["3"]))

        # First entangling layer
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))
        circuit.add(gates.RZ(qubits_list[1], theta=angles_dict["1"]["4"]))
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))

        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))
        circuit.add(gates.RZ(qubits_list[2], theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))

        circuit.add(gates.CNOT(qubits_list[2], qubits_list[3]))
        circuit.add(gates.RZ(qubits_list[3], theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(qubits_list[2], qubits_list[3]))

    return circuit

def load_data(cfg_path):
    """
    Load the configuration data from the JSON file located in the same directory as this script,
    unless an absolute path is provided.
    """
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    return data

def main(qubits_list, device, nshots, debug=False, args=None, input_filename="input.json", number_of_datapoints_output=100000):
    # configure logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    qibo.set_backend("qibolab", platform=device)
    logger.info("Using backend: qibolab (platform=%s)", device)
    
    base_dir = _P(os.path.dirname(os.path.abspath(sys.argv[0])))
    cfg_path = base_dir / "input.json"

    tmp_path = base_dir / (input_filename + "_tmp")
    backup_path = base_dir / (input_filename + "_backup")
    logger.debug("Loading config from %s", cfg_path)

    # Create a backup of the original file
    if cfg_path.exists():
        shutil.copy2(cfg_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

    # Define cleanup function for graceful shutdown
    def cleanup_handler(signum=None, frame=None):
        logger.warning(f"Received termination signal, cleaning up...")
        if tmp_path.exists():
            logger.info(f"Restoring from temporary file {tmp_path}")
            shutil.copy2(tmp_path, cfg_path)
        # Clean up temporary files
        if tmp_path.exists():
            tmp_path.unlink()
        if backup_path.exists():
            backup_path.unlink()
        if signum is not None:
            sys.exit(0)

    # Register signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, cleanup_handler)
    atexit.register(cleanup_handler)

    data = load_data(cfg_path)

    num_qubits = data['args']['num_qubits']
    output_qubit = data['args']['output_qubit']

    # Ensure qubits_list has enough qubits
    if len(qubits_list) < num_qubits:
        logger.error(f"Need at least {num_qubits} qubits, but only {len(qubits_list)} provided")
        return

    out_dir = config.output_dir_for(base_dir / "fix", device)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    # Use physical output qubit
    physical_output_qubit = qubits_list[min(output_qubit, len(qubits_list) - 1)]
    
    # Use Executor for hardware execution
    root_path = out_dir / "executor_tmp"
    
    with Executor.open(
        "qml_executor",
        path=root_path,
        platform=device,
        targets=qubits_list[:num_qubits],
        update=True,
        force=True,
    ) as e:
        logger.info("Using Executor with platform: %s, qubits: %s", device, qubits_list[:num_qubits])
        
        for idx, _ in data['noiseless_experiment_ios'].items():
            # Skip if we've already processed this data point (has sigmoid_expval field)
            if str(idx) in data.get('NQCH', {}) and 'sigmoid_expval' in data['NQCH'][str(idx)]:
                logger.info(f"Skipping data point {idx} - already processed")
                continue
                
            # If we've processed enough points, break
            if processed_count >= number_of_datapoints_output:
                logger.info(f"Reached limit of {number_of_datapoints_output} data points, stopping")
                break

            angles_dict = data['noiseless_experiment_ios'][str(idx)]['angles']
            true_label = data['noiseless_experiment_ios'][str(idx)]['label']
            noiseless_label = data['noiseless_experiment_ios'][str(idx)]['noiseless_label']

            logger.debug("Building circuit for data index %s; true=%s noiseless=%s", idx, true_label, noiseless_label)
            circuit = var_circuit(angles_dict, qubits_list[:num_qubits])
            circuit.add(gates.M(physical_output_qubit))

            start_time = time.time()
            result = circuit(nshots=nshots)
            end_time = time.time()
            duration = end_time - start_time

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
            sigmoid_expval = 1.0 / (1.0 + np.exp(-exp_z))  # sigmoid(<Z>)
            predicted_label = 1 if sigmoid_expval > 0.5 else 0

            # Update the original data with the result
            if 'NQCH' not in data:
                data['NQCH'] = {}
            data['NQCH'][str(idx)] = {
                'sigmoid_expval': float(sigmoid_expval),
                'predicted_label': predicted_label,
                'is_correct': predicted_label == true_label,
                'duration': duration
            }

            # Write to temporary file first, then atomically rename
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            shutil.move(tmp_path, cfg_path)
            logger.debug(f"Updated input file {cfg_path} with results for data point {idx}")

            logger.info("Processed data point %s: predicted=%s true=%s noiseless=%s prob=%.3f",
                        idx, predicted_label, true_label, noiseless_label, sigmoid_expval)  # Fixed variable name
            
            processed_count += 1
        
        report(e.path, e.history)
    
    #################################################################################
    logger.info(f"Processing complete. Total processed data points: {processed_count}")
    #################################################################################

    # Cleanup backup file on successful completion
    if backup_path.exists():
        backup_path.unlink()

    #### Calculate accuracies based on processed results
    # Get all data points that have been processed (have NQCH results)
    evaluated = []
    for idx_str, nqch_result in data.get('NQCH', {}).items():
        if 'sigmoid_expval' in nqch_result and 'predicted_label' in nqch_result:
            # Get corresponding original data
            original_data = data['noiseless_experiment_ios'].get(idx_str, {})
            if original_data:
                evaluated_item = {
                    'qibo_predicted_label': nqch_result['predicted_label'],
                    'label': original_data.get('label'),
                    'noiseless_label': original_data.get('noiseless_label'),
                    'sigmoid_expval': nqch_result['sigmoid_expval']
                }
                evaluated.append(evaluated_item)

    if evaluated:
        qibo_accuracy = sum(c["qibo_predicted_label"] == c["label"] for c in evaluated) / len(evaluated)
        pennylane_vs_qibo_accuracy = sum(c["qibo_predicted_label"] == c["noiseless_label"] for c in evaluated) / len(evaluated)
        original_sim_accuracy = sum(c["label"] == c["noiseless_label"] for c in evaluated) / len(evaluated)

        logger.info("Qibo accuracy (qibo vs true label): %.4f", qibo_accuracy)
        logger.info("PennyLane vs Qibo accuracy (qibo vs noiseless): %.4f", pennylane_vs_qibo_accuracy)
        logger.info(
            "Original noiseless simulation agreement (true vs noiseless): %.4f (expected: %.4f)",
            original_sim_accuracy,
            data.get("accuracy", float("nan")),
        )

        # Store new stats at top-level of JSON structure
        data["NQHC"]["qibo_accuracy"] = qibo_accuracy
        data["NQHC"]["verification_vs_NQHC_accuracy"] = pennylane_vs_qibo_accuracy
        data["NQHC"]["original_sim_accuracy_over_processed_subset"] = original_sim_accuracy
        data["NQHC"]["processed_count"] = len(evaluated)

        # Write full updated data (including qc_configurations) to results.json in out_dir
        results_path = out_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Full results (with accuracies) written to %s", results_path)
    else:
        logger.warning("No data points were processed; no results.json written")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubits_list",
        default=[0, 1, 2],
        type=int,
        nargs='+',
        help="List of physical qubits to use on the device",
    )
    parser.add_argument(
        "--device",
        default="numpy",
        type=str,
        help="Device to use (e.g., 'sinq20' or 'numpy' for local simulation)",
    )
    parser.add_argument(
        "--nshots",
        default=5000,
        type=int,
        help="Number of shots for each circuit execution",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input.json",
        help="Path to the input data file",
    )
    parser.add_argument(
        "--number-datapoints-output",
        type=int,
        default=100000,
        help="Number of data points to output",
    )
    cnf = vars(parser.parse_args())
    main(
        qubits_list=cnf["qubits_list"], 
        device=cnf["device"], 
        nshots=cnf["nshots"], 
        debug=cnf.get("debug", False), 
        args=None, 
        input_filename=cnf.get("input"), 
        number_of_datapoints_output=cnf.get("number_datapoints_output")
    )
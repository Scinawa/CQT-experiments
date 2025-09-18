import argparse
import json
import sys
import pathlib
from pathlib import Path

from qibo import Circuit, gates, set_backend

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # scripts/config.py

from qibocal.auto.execute import Executor
from qibocal.cli.report import report


def bell_circuit():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    return circuit


def run_tomography(targets, device, nshots, root_path):
    if device == "numpy":
        set_backend("numpy")
        platform = None
    else:
        set_backend("qibolab", platform=device)
        platform = device

    results = {}
    data = {
        "targets": targets,
        "nshots": nshots,
        "device": device,
    }

    # Only run Executor if using qibolab
    if platform:
        with Executor.open(
            "myexec",
            path=root_path,
            platform=platform,
            targets=[(targets[0], targets[1])],
            update=True,
            force=True,
        ) as e:
            circuit = bell_circuit()
            start_time = time.time()
            output = e.two_qubit_state_tomography(circuit=circuit, targets=[(targets[0], targets[1])])
            end_time = time.time()
            runtime_seconds = end_time - start_time
            report(e.path, e.history)
            # Save frequencies if available
            if hasattr(output, "frequencies"):
                results["frequencies"] = output.frequencies
                results["runtime"] = runtime_seconds
                results["description"] = f"State tomography on qubits {[targets]}."
    else:
        # Simulate with numpy backend
        start_time = time.time()
        circuit = bell_circuit()
        r = circuit()
        probs = r.probabilities()
        num_bits = 2
        all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
        # Convert probabilities to expected counts, then normalize to frequencies
        freq = {bs: probs[i] for i, bs in enumerate(all_bitstrings)}
        end_time = time.time()
        runtime_seconds = end_time - start_time
        results["frequencies"] = freq
        results["description"] = f"State tomography on numpy backend."

    return data, results


def main(targets, device, nshots):
    scriptname = Path(__file__).stem
    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, results = run_tomography(targets, device, nshots, out_dir)

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
        "--targets",
        default=[13, 8],
        type=lambda s: [int(i) for i in s.split(",")],
        help="Comma-separated list of two target qubits, e.g. 13,8",
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
    main(args.targets, args.device, args.nshots)

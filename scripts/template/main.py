
import os
import pathlib
import argparse
from qibo import set_backend, Circuit, gates
import json
import sys



sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # scripts/config.py


def circuit():

    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(*[0, 1]))

    return circuit

def main(qubit_groups, device, nshots):
    if device == "numpy":
        set_backend("numpy")
    else:
        set_backend("qibolab", platform=device)
    
    c = circuit()
    r= c(nshots=nshots)

    freq = r.frequencies()
    print(f"Frequencies: {freq}")

    out_dir = config.output_dir_for(__file__, args.device)
    

    # os.makedirs(out_dir, exist_ok=True)
    # with open(os.path.join(out_dir, "results.json"), "w") as f:
    #     json.dump(results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubit_groups",
        default=[[0, 1, 3, 4]],
        type=list,
        help="Target qubits, last qubit used as ancilla",
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "nqch-sim", "sinq20"],
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
    main(**args)
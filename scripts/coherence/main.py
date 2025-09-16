import os
import pathlib

# os.environ["QIBOLAB_PLATFORMS"] = (pathlib.Path(__file__).parent / "qibolab_platforms_nqch").as_posix()

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

import argparse
import json
import sys
from pathlib import Path as _P
import time

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

def main(device):

    results = dict()
    data = dict()

    results["t1"] = {}
    results["t2"] = {}
    
    data["device"] = device

    _tmp_runtimes = []

    root_path = "coherence"
    platform = device

    qb_array1 = [0, 2, 4, 8, 12, 6, 10, 14, 17, 16, 19]
    qb_array2 = [1, 3, 7, 5, 9, 13, 18, 11, 15]

    data["qubits"] = [qb_array1, qb_array1]

    params_t1 = {
        "delay_before_readout_end": 100000,
        "delay_before_readout_start": 20,
        "delay_before_readout_step": 2000
    }
    params_t2_ramsey = {
        "delay_between_pulses_start": 20,
        "delay_between_pulses_end": 20000,
        "delay_between_pulses_step": 100,
        "detuning": 200000
    }
     
    with Executor.open(
        "myexec",
        path=root_path,
        platform=platform,
        update=True,
        force=True,
    ) as e:
        
        start_time = time.time()
        e.t1(parameters=params_t1, targets=qb_array1)
        e.t1(parameters=params_t1, targets=qb_array2)
        results_exp: dict = e.ramsey(parameters=params_t2_ramsey, targets=qb_array1)._results.t2
        results_exp.update(e.ramsey(parameters=params_t2_ramsey, targets=qb_array2)._results.t2)
        end_time = time.time()
        _tmp_runtimes.append(end_time - start_time)

        report(e.path, e.history)

        for qubit in range(1, 21):
            qubit_id = qubit - 1
            t1_val = float(e.platform.calibration.single_qubits[qubit_id].t1[0]) / 1e3
            t2_val = float(results_exp[qubit_id][0]) / 1e3

            t1=0 if t1_val < 0 or t1_val > 100 else t1_val
            t2=0 if t2_val < 0 or t2_val > 100 else t2_val

            results["t1"][f"{qubit_id}"] = t1
            results["t2"][f"{qubit_id}"] = t2

    runtime_seconds = sum(_tmp_runtimes) / len(_tmp_runtimes) if _tmp_runtimes else 0.0
    results['runtime']= f"{runtime_seconds:.5f} seconds."
    results['description']= f"T1 and T2 experiments performed on {device} backend. \n The thermalization and coherence times are computed with this routine."

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
        "--device",
        choices=["sinq20"],
        default="sinq20",
        type=str,
        help="Device to use",
    )

    args = vars(parser.parse_args())
    main(**args)

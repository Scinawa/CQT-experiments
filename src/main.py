import argparse
from multiprocessing import context
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import logging
import os
import json
import numpy as np
import fillers as fl
import sys
import plots as pl
import config
import pdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def setup_argument_parser():
    """Set up the command line argument parser."""
    parser = argparse.ArgumentParser(description="Generate a quantum benchmark report.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/",
        help="Base directory for data and output.",
    )

    parser.add_argument(
        "--experiment-right",
        type=str,
        default="9848c933bfcafbb8f81c940f504b893a2fa6ac23",
        help="Directory containing the experiment data.",
    )
    parser.add_argument(
        "--experiment-left",
        type=str,
        default="numpy",
        help="Directory containing the experiment data.",
    )

    # Plot toggles (default: True). Use --no-<flag> to disable.
    parser.add_argument("--no-t1-plot", dest="t1_plot", action="store_false")
    parser.add_argument(
        "--no-mermin-5-plot", dest="mermin_5_plot", action="store_false"
    )
    parser.add_argument(
        "--no-reuploading-plot", dest="reuploading_plot", action="store_false"
    )
    parser.add_argument(
        "--no-grover2q-plot", dest="grover2q_plot", action="store_false"
    )
    parser.add_argument(
        "--no-grover3q-plot", dest="grover3q_plot", action="store_false"
    )

    parser.add_argument("--no-ghz-plot", dest="ghz_plot", action="store_false")
    parser.add_argument(
        "--no-process-tomography-plot",
        dest="process_tomography_plot",
        action="store_false",
    )
    parser.add_argument(
        "--no-qft-plot",
        dest="qft_plot",
        action="store_false",
    )
    parser.add_argument(
        "--no-tomography-plot", dest="tomography_plot", action="store_false"
    )
    parser.add_argument(
        "--no-reuploading-classifier-plot",
        dest="reuploading_classifier_plot",
        action="store_false",
    )
    parser.add_argument(
        "--no-yeast-plot-4q", dest="yeast_plot_4q", action="store_false"
    )
    parser.add_argument(
        "--no-yeast-plot-3q", dest="yeast_plot_3q", action="store_false"
    )
    parser.add_argument(
        "--no-statlog-plot-4q", dest="statlog_plot_4q", action="store_false"
    )
    parser.add_argument(
        "--no-statlog-plot-3q", dest="statlog_plot_3q", action="store_false"
    )
    parser.add_argument(
        "--no-amplitude-encoding-plot",
        dest="amplitude_encoding_plot",
        action="store_false",
    )
    return parser


def add_stat_changes(current, baseline):
    """
    Returns a dict with average, min, max, median and their changes vs baseline.
    """

    def get_change(curr, base):
        if curr is None or base is None:
            return None
        try:
            diff = float(curr) - float(base)
            if base == 0:
                return None
            percent = (diff / float(base)) * 100
            if percent > 0:
                return f"(+{percent:.2f}\\%)"
            elif percent < 0:
                return f"(-{percent:.2f}\\%)"
            else:
                return "-"
            # return f"{diff:+.4g} ({percent:+.2f}\\%)"
        except Exception as e:
            print(f"Error calculating change: {e}")
            sys.exit(1)
            return None

    result = {}
    for key in ["average", "min", "max", "median"]:
        curr_val = current.get(key)
        base_val = baseline.get(key)
        result[key] = curr_val
        result[f"{key}_change"] = get_change(curr_val, base_val)
    # print(f"Stat fidelity changes: {result}")
    return result


def get_experiment_info(experiment_path, experiment_dir):
    """
    Extract description and duration from experiment JSON files.

    Args:
        experiment_path (str): Path to experiment (left or right)
        experiment_dir (str): Directory for experiment type (e.g., 'mermin', 'grover2q')

    Returns:
        tuple: (description, duration)
    """
    description = "No description available"
    duration = "Unknown duration"

    # Try to get description from results.json
    results_path = os.path.join("data", experiment_dir, experiment_path, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results_data = json.load(f)
                if "description" in results_data:
                    description = results_data["description"]
        except Exception as e:
            logging.warning(f"Error reading description from {results_path}: {e}")

    # Try to get duration from report.json
    report_path = os.path.join("data", experiment_dir, experiment_path, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                report_data = json.load(f)
                if "duration" in report_data:
                    duration = report_data["duration"]
        except Exception as e:
            logging.warning(f"Error reading duration from {report_path}: {e}")

    return description, duration


def prepare_template_context(cfg):
    """
    Prepare a complete template context for the benchmarking report.
    """

    logging.info("Preparing context for full benchmarking report.")

    base_path = Path(cfg.base_dir)

    # Load experiment metadata from mega.json
    meta_json_path = base_path / cfg.experiment_left / "sinq20" / "meta.json"
    with open(meta_json_path, "r") as f:
        meta_data = json.load(f)
    logging.info("Loaded experiment metadata from %s", meta_json_path)

    ######### Fidelity statistics and changes
    try:
        stat_fidelity = fl.get_stat_fidelity(cfg.experiment_left + "/sinq20")
        stat_fidelity_baseline = fl.get_stat_fidelity(cfg.experiment_right + "/sinq20")
        stat_fidelity_with_improvement = add_stat_changes(
            stat_fidelity, stat_fidelity_baseline
        )
        logging.info("Prepared stat_fidelity and stat_fidelity_with_improvement")
    except Exception as e:
        logging.error(f"Error preparing fidelity statistics: {e}")
        stat_fidelity = {}
        stat_fidelity_baseline = {}
        stat_fidelity_with_improvement = {}

    ##########Pulse Fidelity statistics and changes
    try:
        stat_pulse_fidelity = fl.get_stat_pulse_fidelity(
            cfg.experiment_left + "/sinq20"
        )
        stat_pulse_fidelity_baseline = fl.get_stat_pulse_fidelity(
            cfg.experiment_right + "/sinq20"
        )
        stat_pulse_fidelity_with_improvement = add_stat_changes(
            stat_pulse_fidelity, stat_pulse_fidelity_baseline
        )
        logging.info(
            "Prepared stat_pulse_fidelity and stat_pulse_fidelity_with_improvement"
        )
    except Exception as e:
        logging.error(f"Error preparing pulse fidelity statistics: {e}")
        stat_pulse_fidelity = {}
        stat_pulse_fidelity_baseline = {}
        stat_pulse_fidelity_with_improvement = {}

    ######### T1 statistics and changes
    stat_t1 = fl.get_stat_t12(cfg.experiment_left + "/sinq20", "t1")
    stat_t1_baseline = fl.get_stat_t12(cfg.experiment_right + "/sinq20", "t1")
    stat_t1_with_improvement = add_stat_changes(stat_t1, stat_t1_baseline)
    logging.info("Prepared stat_t1 and stat_t1_with_improvement")

    ######### T2 statistics and changes
    stat_t2 = fl.get_stat_t12(cfg.experiment_left + "/sinq20", "t2")
    stat_t2_baseline = fl.get_stat_t12(cfg.experiment_right + "/sinq20", "t2")
    stat_t2_with_improvement = add_stat_changes(stat_t2, stat_t2_baseline)
    logging.info("Prepared stat_t2 and stat_t2_with_improvement")

    context = {
        "experiment_name": meta_data.get("title", "Unknown Title"),
        "platform": meta_data.get("platform", "Unknown Platform"),
        #
        "start_time": meta_data.get("start-time", "Unknown Start Time"),
        "end_time": meta_data.get("start-time", "Unknown Start Time"),
        #
        "report_of_changes": " ", #"\\textcolor{green}{Additional data of changes (from software).}",
        #
        "stat_fidelity": stat_fidelity_with_improvement,
        "stat_fidelity_baseline": stat_fidelity_baseline,
        #
        "stat_pulse_fidelity": stat_pulse_fidelity_with_improvement,
        "stat_pulse_fidelity_baseline": stat_pulse_fidelity_baseline,
        #
        "stat_t1": stat_t1_with_improvement,
        "stat_t1_baseline": stat_t1_baseline,
        #
        "stat_t2": stat_t2_with_improvement,
        "stat_t2_baseline": stat_t2_baseline,
        #
        "version": fl.context_version(
            os.path.join(
                "data", "version_extractor", cfg.experiment_left, "results.json"
            )
        ),
        "version_baseline": fl.context_version(
            os.path.join(
                "data", "version_extractor", cfg.experiment_right, "results.json"
            )
        ),
    }
    #
    try:
        context["fidelity"] = fl.context_fidelity(cfg.experiment_left + "/sinq20")
        context["fidelity_baseline"] = fl.context_fidelity(
            cfg.experiment_right + "/sinq20"
        )
    except Exception as e:
        logging.error(f"Error preparing fidelity context: {e}")
        context["fidelity"] = {}
        context["fidelity_baseline"] = {}
        #
    try:
        context["plot_exp"] = pl.plot_fidelity_graph(
            cfg.experiment_left + "/sinq20", config.connectivity, config.pos
        )
        context["plot_baseline"] = pl.plot_fidelity_graph(
            cfg.experiment_right + "/sinq20", config.connectivity, config.pos
        )
    except Exception as e:
        logging.error(f"Error preparing fidelity plots: {e}")
        context["plot_exp"] = "placeholder.png"
        context["plot_baseline"] = "placeholder.png"
        #
        # "plot_chevron_swap_0": pl.plot_chevron_swap_coupler(
        #     qubit_number=0,
        #     data_dir="data/DEMODATA/",
        #     output_path="build/",
        # ),
        # #
        # "plot_swap_coupler": pl.plot_swap_coupler(
        #     qubit_number=0,
        #     data_dir="data/DEMODATA/",
        #     output_path="build/",
        # ),

    logging.info("Basic context dictionary prepared")

    # # Add additional plots if needed
    # context["grid_coupler_is_set"] = False
    # grid_coupler_plots = pl.prepare_grid_coupler(
    #     max_number=2,
    #     data_dir="data/DEMODATA",
    #     baseline_dir="data/DEMODATA",
    #     output_path="build/",
    # )
    # context["plot_grid_coupler"] = grid_coupler_plots
    # logging.info("Added grid_coupler plots to context")

    # # Add additional chevron_swap_coupler plots if needed
    # context["chevron_swap_coupler_is_set"] = False
    # chevron_swap_coupler_plots = pl.prepare_grid_chevron_swap_coupler(
    #     max_number=2,
    #     data_dir="data/DEMODATA",
    #     baseline_dir="data/DEMODATA",
    #     output_path="build/",
    # )
    # context["plot_chevron_swap_coupler"] = chevron_swap_coupler_plots
    # logging.info("Added chevron_swap_coupler plots to context")

    # ######### T1 PLOT
    if cfg.t1_plot == True:
        try:
            context["t1_plot_is_set"] = True
            t1_plot = pl.prepare_grid_t1_plts(
                max_number=2,
                data_dir="data/DEMODATA",
                output_path="build/",
            )
            context["plot_grid_t1"] = t1_plot
            logging.info("Added T1 plots to context")
        except Exception as e:
            logging.error(f"Error adding T1 plots: {e}")
            context["t1_plot_is_set"] = False
    else:
        context["t1_plot_is_set"] = False
        pass

    ######### MERMIN TABLE
    maximum_mermin = fl.get_maximum_mermin(
        base_path / "mermin" / cfg.experiment_left, "results.json"
    )
    maximum_mermin_baseline = fl.get_maximum_mermin(
        base_path / "mermin" / cfg.experiment_right, "results.json"
    )
    context["mermin_maximum"] = maximum_mermin
    context["mermin_maximum_baseline"] = maximum_mermin_baseline

    ######### MERMIN PLOTS
    if cfg.mermin_5_plot == True:
        try:
            context["mermin_5_plot_is_set"] = True
            context["plot_mermin"] = pl.mermin_plot(
                raw_data=os.path.join(
                    "data", "mermin", cfg.experiment_left, "results.json"
                ),
                expname=f"mermin_{cfg.experiment_left}",
                output_path="build/",
            )
            context["plot_mermin_baseline"] = mermin_5_plot_baseline = pl.mermin_plot(
                raw_data=os.path.join(
                    "data", "mermin", cfg.experiment_right, "results.json"
                ),
                expname=f"mermin_{cfg.experiment_right}",
                output_path="build/",
            )
            logging.info("Added Mermin 5Q plots to context")
        except Exception as e:
            logging.error(f"Error adding Mermin 5Q plots: {e}")
            context["mermin_5_plot_is_set"] = False
    else:
        context["mermin_5_plot_is_set"] = False
        pass

    ######### GROVER 2q PLOTS
    if cfg.grover2q_plot == True:
        try:
            context["grover2q_plot_is_set"] = True
            context["plot_grover2q"] = pl.plot_grover(
                raw_data=os.path.join(
                    "data", "grover2q", cfg.experiment_left, "results.json"
                ),
                expname=f"grover2q_{cfg.experiment_left}",
                output_path="build/",
            )
            context["plot_grover2q_baseline"] = pl.plot_grover(
                raw_data=os.path.join(
                    "data", "grover2q", cfg.experiment_right, "results.json"
                ),
                expname=f"grover2q_{cfg.experiment_right}",
                output_path="build/",
            )
            logging.info("Added Grover 2Q plots to context")
        except Exception as e:
            logging.error(f"Error adding Grover 2Q plots: {e}")
            context["grover2q_plot_is_set"] = False
    else:
        context["grover2q_plot_is_set"] = False
        pass

    ######### GROVER 3q PLOTS
    if cfg.grover3q_plot == True:
        try:
            context["grover3q_description"] = fl.extract_description(
                os.path.join("data", "grover3q", cfg.experiment_left, "results.json")
            )
            context["grover3q_runtime_left"] = fl.extract_runtime(
                os.path.join("data", "grover3q", cfg.experiment_left, "results.json")
            )
            context["grover3q_runtime_right"] = fl.extract_runtime(
                os.path.join("data", "grover3q", cfg.experiment_right, "results.json")
            )
            context["grover3q_plot_is_set"] = True
            context["plot_grover3q"] = pl.plot_grover(
                raw_data=os.path.join(
                    "data", "grover3q", cfg.experiment_left, "results.json"
                ),
                expname=f"grover3q_{cfg.experiment_left}",
                output_path="build/",
            )
            context["plot_grover3q_baseline"] = pl.plot_grover(
                raw_data=os.path.join(
                    "data", "grover3q", cfg.experiment_right, "results.json"
                ),
                expname=f"grover3q_{cfg.experiment_right}",
                output_path="build/",
            )
            logging.info("Added Grover 3Q plots to context")
        except Exception as e:
            logging.error(f"Error adding Grover 3Q plots: {e}")
            context["grover3q_plot_is_set"] = False
    else:
        context["grover3q_plot_is_set"] = False
        pass

    ######### GHZ PLOTS
    if cfg.ghz_plot == True:
        try:
            context["ghz_plot_is_set"] = True
            context["ghz_description"] = fl.extract_description(
                os.path.join("data", "GHZ", cfg.experiment_left, "results.json")
            )
            context["ghz_runtime_left"] = fl.extract_runtime(
                os.path.join("data", "GHZ", cfg.experiment_left, "results.json")
            )
            context["ghz_runtime_right"] = fl.extract_runtime(
                os.path.join("data", "GHZ", cfg.experiment_right, "results.json")
            )

            context["plot_ghz"] = pl.plot_ghz(
                raw_data=os.path.join(
                    "data", "GHZ", cfg.experiment_left, "results.json"
                ),
                experiment_name=cfg.experiment_left,
                output_path=os.path.join("build", "GHZ", cfg.experiment_left),
            )
            context["plot_ghz_baseline"] = pl.plot_ghz(
                raw_data=os.path.join(
                    "data", "GHZ", cfg.experiment_right, "results.json"
                ),
                experiment_name=cfg.experiment_right,
                output_path=os.path.join("build", "GHZ", cfg.experiment_right),
            )
            logging.info("Added GHZ plots to context")
        except Exception as e:
            logging.error(f"Error adding GHZ plots: {e}")
            context["ghz_plot_is_set"] = False
    else:
        logging.info("GHZ plot is not set, skipping...")
        context["ghz_plot_is_set"] = False
        pass

    ######### PROCESS TOMOGRAPHY PLOTS
    if cfg.process_tomography_plot == True:
        try:
            context["process_tomography_plot_is_set"] = True
            context["plot_process_tomography"] = pl.plot_process_tomography(
                # raw_data=os.path.join(
                # "data", "process_tomography", cfg.experiment_left, "results.json"
                # ),
                expname=cfg.experiment_left,  # f"process_tomography_{cfg.experiment_left}",
                output_path=os.path.join(
                    "build", "process_tomography", cfg.experiment_left
                ),
            )
            context["plot_process_tomography_baseline"] = pl.plot_process_tomography(
                # raw_data=os.path.join(
                # "data", "process_tomography", cfg.experiment_right, "results.json"
                # ),
                expname=cfg.experiment_right,  # f"process_tomography_{cfg.experiment_right}",
                output_path=os.path.join(
                    "build", "process_tomography", cfg.experiment_right
                ),
            )
            logging.info("Added Process Tomography plots to context")
        except Exception as e:
            logging.error(f"Error adding Process Tomography plots: {e}")
            context["process_tomography_plot_is_set"] = False
    else:
        logging.info("Process Tomography plot is not set, skipping...")
        context["process_tomography_plot_is_set"] = False
        pass

    ######### TOMOGRAPHY PLOTS
    if cfg.tomography_plot == True:
        try:
            context["tomography_plot_is_set"] = True
            context["plot_tomography"] = pl.plot_tomography(
                raw_data=os.path.join(
                    "data", "tomography", cfg.experiment_left, "results.json"
                ),
                output_path=os.path.join("build", "tomography", cfg.experiment_left),
            )
            context["plot_tomography_baseline"] = pl.plot_tomography(
                raw_data=os.path.join(
                    "data", "tomography", cfg.experiment_right, "results.json"
                ),
                output_path=os.path.join("build", "tomography", cfg.experiment_right),
            )
            logging.info("Added Tomography plots to context")
        except Exception as e:
            logging.error(f"Error adding Tomography plots: {e}")
            context["tomography_is_set"] = False
    else:
        logging.info("Tomography plot is not set, skipping...")
        context["tomography_plot_is_set"] = False

    ######### REUPLOADING CLASSIFIER PLOTS
    if cfg.reuploading_classifier_plot == True:
        try:
            context["reuploading_classifier_description"] = fl.extract_description(
                os.path.join(
                    "data",
                    "reuploading_classifier",
                    cfg.experiment_left,
                    "results.json",
                )
            )
            context["reuploading_classifier_runtime_left"] = fl.extract_runtime(
                os.path.join(
                    "data",
                    "reuploading_classifier",
                    cfg.experiment_left,
                    "results.json",
                )
            )
            context["reuploading_classifier_runtime_right"] = fl.extract_runtime(
                os.path.join(
                    "data",
                    "reuploading_classifier",
                    cfg.experiment_right,
                    "results.json",
                )
            )

            context["reuploading_classifier_plot_is_set"] = True
            context["plot_reuploading_classifier"] = pl.plot_reuploading_classifier(
                raw_data=os.path.join(
                    "data",
                    "reuploading_classifier",
                    cfg.experiment_left,
                    "results.json",
                ),
                exp_name=cfg.experiment_left,
                output_path=os.path.join(
                    "build", "reuploading_classifier", cfg.experiment_left
                ),
            )
            context["plot_reuploading_classifier_baseline"] = (
                pl.plot_reuploading_classifier(
                    raw_data=os.path.join(
                        "data",
                        "reuploading_classifier",
                        cfg.experiment_right,
                        "results.json",
                    ),
                    exp_name=cfg.experiment_right,
                    output_path=os.path.join(
                        "build", "reuploading_classifier", cfg.experiment_right
                    ),
                )
            )
            logging.info("Added Reuploading Classifier plots to context")
        except Exception as e:
            logging.error(f"Error adding Reuploading Classifier plots: {e}")
            context["reuploading_classifier_is_set"] = False
    else:
        logging.info("Reuploading Classifier plot is not set, skipping...")
        context["reuploading_classifier_plot_is_set"] = False
        pass

    ######### QFT PLOTS
    if cfg.qft_plot == True:
        try:
            context["QFT_description"] = fl.extract_description(
                os.path.join("data", "QFT", cfg.experiment_left, "results.json")
            )
            context["QFT_runtime_left"] = fl.extract_runtime(
                os.path.join("data", "QFT", cfg.experiment_left, "results.json")
            )
            context["QFT_runtime_right"] = fl.extract_runtime(
                os.path.join("data", "QFT", cfg.experiment_right, "results.json")
            )

            context["qft_plot_is_set"] = True
            context["plot_qft"] = pl.plot_QFT(
                raw_data=os.path.join(
                    "data", "QFT", cfg.experiment_left, "results.json"
                ),
                expname=f"QFT_{cfg.experiment_left}",
                output_path=os.path.join("build", "QFT", cfg.experiment_left),
            )
            context["plot_qft_baseline"] = pl.plot_QFT(
                raw_data=os.path.join(
                    "data", "QFT", cfg.experiment_right, "results.json"
                ),
                expname=f"QFT_{cfg.experiment_right}",
                output_path=os.path.join("build", "QFT", cfg.experiment_right),
            )
            logging.info("Added QFT plots to context")
        except Exception as e:
            logging.error(f"Error adding QFT plots: {e}")
            context["qft_plot_is_set"] = False
    else:
        logging.info("QFT plot is not set, skipping...")
        context["qft_plot_is_set"] = False
        pass

    ######### YEAST CLASSIFICATION PLOTS 4Q
    if cfg.yeast_plot_4q == True:
        try:
            context["qml_4Q_yeast_description"] = fl.extract_description(
                os.path.join(
                    "data", "qml_4Q_yeast", cfg.experiment_left, "results.json"
                )
            )
            context["qml_4Q_yeast_runtime_left"] = fl.extract_runtime(
                os.path.join(
                    "data", "qml_4Q_yeast", cfg.experiment_left, "results.json"
                )
            )
            context["qml_4Q_yeast_runtime_right"] = fl.extract_runtime(
                os.path.join(
                    "data", "qml_4Q_yeast", cfg.experiment_right, "results.json"
                )
            )

            context["yeast_classification_4q_plot_is_set"] = True
            context["plot_yeast_4q"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_4Q_yeast", cfg.experiment_left, "results.json"
                ),
                expname=f"4q_yeast_{cfg.experiment_left}",
                output_path=os.path.join("build", "yeast", cfg.experiment_left),
            )
            context["plot_yeast_4q_baseline"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_4Q_yeast", cfg.experiment_right, "results.json"
                ),
                expname=f"4q_yeast_{cfg.experiment_right}",
                output_path=os.path.join("build", "yeast", cfg.experiment_right),
            )
            logging.info("Added Yeast classification 4q plots to context")
        except Exception as e:
            logging.error(f"Error adding Yeast classification 4q plots: {e}")
            context["yeast_classification_4q_plot_is_set"] = False
    else:
        logging.info("Yeast classification 4q plot is not set, skipping...")
        context["yeast_classification_4q_plot_is_set"] = False
        pass

    ######### YEAST CLASSIFICATION PLOTS 3Q
    if cfg.yeast_plot_3q == True:
        try:
            context["yeast_classification_3q_plot_is_set"] = True
            context["yeast_3q_accuracy_right"] = fl.get_qml_accuracy(
                os.path.join(
                    "data", "qml_3Q_yeast", cfg.experiment_left, "results.json"
                )
            )
            context["yeast_3q_accuracy_left"] = fl.get_qml_accuracy(
                os.path.join(
                    "data", "qml_3Q_yeast", cfg.experiment_left, "results.json"
                )
            )
            context["plot_yeast_3q"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_3Q_yeast", cfg.experiment_left, "results.json"
                ),
                expname=f"3q_yeast_{cfg.experiment_left}",
                output_path=os.path.join("build", "yeast", cfg.experiment_left),
            )
            context["plot_yeast_3q_baseline"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_3Q_yeast", cfg.experiment_right, "results.json"
                ),
                expname=f"3q_yeast_{cfg.experiment_right}",
                output_path=os.path.join("build", "yeast", cfg.experiment_right),
            )
            logging.info("Added Yeast classification 3q plots to context")
        except Exception as e:
            logging.error(f"Error adding Yeast classification 3q plots: {e}")
            context["yeast_classification_3q_plot_is_set"] = False
    else:
        logging.info("Yeast classification 3q plot is not set, skipping...")
        context["yeast_classification_3q_plot_is_set"] = False
        pass

    ######### STATLOG CLASSIFICATION PLOTS 4Q
    if cfg.statlog_plot_4q == True:
        try:
            context["statlog_classification_4q_is_set"] = True
            context["plot_statlog_4q"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_4Q_statlog", cfg.experiment_left, "results.json"
                ),
                expname=f"4q_statlog_{cfg.experiment_left}",
                output_path=os.path.join("build", "statlog", cfg.experiment_left),
            )
            context["plot_statlog_4q_baseline"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_4Q_statlog", cfg.experiment_right, "results.json"
                ),
                expname=f"4q_statlog_{cfg.experiment_right}",
                output_path=os.path.join("build", "statlog", cfg.experiment_right),
            )
            logging.info("Added StatLog classification 4q plots to context")
        except Exception as e:
            logging.error(f"Error adding StatLog classification 4q plots: {e}")
            context["statlog_classification_4q_is_set"] = False
    else:
        logging.info("StatLog classification 4q plot is not set, skipping...")
        context["statlog_classification_4q_plot_is_set"] = False
        pass

    ######### STATLOG CLASSIFICATION PLOTS 3Q
    if cfg.statlog_plot_3q == True:
        try:
            context["statlog_classification_3q_plot_is_set"] = True
            context["statlog_3q_accuracy_right"] = fl.get_qml_accuracy(
                os.path.join(
                    "data", "qml_3Q_statlog", cfg.experiment_left, "results.json"
                )
            )
            context["statlog_3q_accuracy_left"] = fl.get_qml_accuracy(
                os.path.join(
                    "data", "qml_3Q_statlog", cfg.experiment_left, "results.json"
                )
            )
            context["plot_statlog_3q"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_3Q_statlog", cfg.experiment_left, "results.json"
                ),
                expname=f"3q_statlog_{cfg.experiment_left}",
                output_path=os.path.join("build", "statlog", cfg.experiment_left),
            )
            context["plot_statlog_3q_baseline"] = pl.plot_qml(
                raw_data=os.path.join(
                    "data", "qml_3Q_statlog", cfg.experiment_right, "results.json"
                ),
                expname=f"3q_statlog_{cfg.experiment_right}",
                output_path=os.path.join("build", "statlog", cfg.experiment_right),
            )
            logging.info("Added StatLog classification 3q plots to context")
        except Exception as e:
            logging.error(f"Error adding StatLog classification 3q plots: {e}")
            context["statlog_classification_3q_plot_is_set"] = False
    else:
        logging.info("StatLog classification 3q plot is not set, skipping...")
        context["statlog_classification_3q_plot_is_set"] = False
        pass

    ######### AMPLITUDE ENCODING PLOTS
    if cfg.amplitude_encoding_plot == True:
        try:
            context["amplitude_encoding_plot_is_set"] = True
            context["plot_amplitude_encoding"] = pl.plot_amplitude_encoding(
                raw_data=os.path.join(
                    "data", "amplitude_encoding", cfg.experiment_left, "results.json"
                ),
                expname=f"amplitude_encoding_{cfg.experiment_left}",
                output_path=os.path.join(
                    "build", "amplitude_encoding", cfg.experiment_left
                ),
            )
            context["plot_amplitude_encoding_baseline"] = pl.plot_amplitude_encoding(
                raw_data=os.path.join(
                    "data", "amplitude_encoding", cfg.experiment_right, "results.json"
                ),
                expname=f"amplitude_encoding_{cfg.experiment_right}",
                output_path=os.path.join(
                    "build", "amplitude_encoding", cfg.experiment_right
                ),
            )
            logging.info("Added Amplitude Encoding plots to context")
        except Exception as e:
            logging.error(f"Error adding Amplitude Encoding plots: {e}")
            context["amplitude_encoding_plot_is_set"] = False
    else:
        logging.info("Amplitude Encoding plot is not set, skipping...")
        context["amplitude_encoding_plot_is_set"] = False
        pass

    # Define experiment mapping (experiment directory → context key prefix)
    experiment_mapping = {
        "mermin": "mermin",
        "grover2q": "grover2q",
        "grover3q": "grover3q",
        "GHZ": "ghz",
        "tomography": "tomography",
        "process_tomography": "process_tomography",
        "reuploading_classifier": "reuploading_classifier",
        "QFT": "qft",
        "qml_4Q_yeast": "yeast_4q",
        "qml_3Q_yeast": "yeast_3q",
        "qml_4Q_statlog": "statlog_4q",
        "qml_3Q_statlog": "statlog_3q",
        "amplitude_encoding": "amplitude_encoding",
    }

    # Extract experiment info and add to context
    for exp_dir, context_key in experiment_mapping.items():
        left_desc, left_duration = get_experiment_info(cfg.experiment_left, exp_dir)
        right_desc, right_duration = get_experiment_info(cfg.experiment_right, exp_dir)

        # Add to context - use the left experiment description for the main description
        context[f"{context_key}_description"] = left_desc
        context[f"{context_key}_duration_left"] = left_duration
        context[f"{context_key}_duration_right"] = right_duration

        logging.info(f"Added description and duration for {exp_dir}")

    return context


def render_and_save_report(context, args):
    """
    Render the LaTeX template and save the generated report.

    Args:
        context: Template context dictionary.
        args: Parsed command line arguments.

    Returns:
        Path to the generated LaTeX file.
    """
    logging.info("Rendering LaTeX template...")
    # Setup Jinja2 environment and load template
    template_dir = Path("src/templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.j2")

    # Render template with context
    rendered_content = template.render(context)

    # Ensure build directory exists
    build_dir = Path(".")
    build_dir.mkdir(exist_ok=True)

    # Write rendered LaTeX file in the build directory using os.path.join
    output_file = os.path.join(build_dir, "report.tex")
    with open(output_file, "w") as f:
        f.write(rendered_content)
        return output_file

    raise RuntimeError("Failed to render the report template.")


def main():
    """
    Main function that orchestrates the quantum benchmark report generation process.
    """
    logging.info("Starting report generation process...")

    # Step 1: Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Step 2: Load and validate experiment data (dummy call for now, can be removed if not needed for experiment_name)
    # For simplicity, we are not loading actual experiment data as per the request
    # to only show library versions.
    # If experiment_left is purely for the title, this can be simplified further.
    logging.info(
        f"Targeting experiment directory for report title: {args.experiment_left}"
    )

    # Step 3: Prepare template context with all required data
    context = prepare_template_context(args)

    # Step 4: Render and save the LaTeX report
    output_file = render_and_save_report(context, args)

    # Step 5: Report completion
    logging.info(f"Report generated at: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())

import quantus
import pathlib
from datetime import datetime
import argparse
import time
import json
import logging
import os
import glob
import torch.distributed as dist
import torch

import os
import argparse
import torch
from datetime import datetime

from metaquantus import MetaEvaluation, MetaEvaluationBenchmarking

from src.helpers.experiments.meta_eval.configs import (
    setup_metrics_meta,
    setup_tasks,
    setup_xai_methods_zennit,
    setup_xai_methods_captum,
    setup_test_suite_gef,
    DATASETS,
    XAI_METHODS_ALL,
)

# Check if running in a distributed setting.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
is_distributed = "WORLD_SIZE" in os.environ

# Set up argument parser.
parser = argparse.ArgumentParser(description="Process the experiment.")
parser.add_argument("--fname", type=str, required=True, help="Base experiment name.")
parser.add_argument(
    "--path_assets",
    type=str,
    required=False,
    help="Base path assets.",
    default="../assets/",
)
parser.add_argument(
    "--path_results",
    type=str,
    required=False,
    help="Base path assets.",
    default="MNIST",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=False,
    help="Base dataset_name.",
    default="MNIST",
)
parser.add_argument(
    "--T", type=int, required=False, help="Base value of T.", default="10"
)
parser.add_argument(
    "--K", type=int, required=False, help="Base value of K.", default="10"
)
parser.add_argument(
    "--Z", type=int, required=False, help="Base value of Z.", default="5"
)
parser.add_argument(
    "--M", type=int, required=False, help="Base value of M.", default="5"
)
parser.add_argument("--xai_round", type=str, required=True, help="Base XAI group.")

# Parse arguments.
args = parser.parse_args()
path_assets = args.path_assets
path_results = args.path_results
dataset_name = args.dataset_name
xai_round = args.xai_round
fname = args.fname
gef_kwargs = {"T": args.T, "K": args.K, "M": args.M, "Z": args.Z}


# Check if running in a distributed setting.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
is_distributed = "WORLD_SIZE" in os.environ

# Set up argument parser.
parser = argparse.ArgumentParser(description="Process the experiment.")
parser.add_argument("--fname", type=str, required=True, help="Base experiment name.")
parser.add_argument(
    "--path_assets",
    type=str,
    required=False,
    help="Base path assets.",
    default="../assets/",
)
parser.add_argument(
    "--path_results",
    type=str,
    required=False,
    help="Base path results.",
    default="../cluster_benchmarking/",
)
parser.add_argument(
    "--folder",
    type=str,
    required=False,
    help="Base folder results.",
    default="benchmarking/",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=False,
    help="Base dataset_name.",
    default="MNIST",
)
parser.add_argument(
    "--T", type=int, required=False, help="Base value of T.", default="10"
)
parser.add_argument(
    "--K", type=int, required=False, help="Base value of K.", default="10"
)
parser.add_argument(
    "--Z", type=int, required=False, help="Base value of Z.", default="5"
)
parser.add_argument(
    "--M", type=int, required=False, help="Base value of M.", default="5"
)
parser.add_argument("--xai_round", type=str, required=True, help="Base XAI group.")

# Parse arguments.
args = parser.parse_args()
path_assets = args.path_assets
path_results = args.path_results
folder = args.folder
dataset_name = args.dataset_name
xai_round = args.xai_round
fname = args.fname
gef_kwargs = {"T": args.T, "K": args.K, "M": args.M, "Z": args.Z}


def consolidate_meta_evaluation_outputs(
    base_path, output_pattern, final_output, default=None
):
    """
    Gather all output files matching a pattern and consolidate them into a single file using JSON.
    Adds error handling to provide more informative messages.
    """
    all_files = glob.glob(os.path.join(base_path, output_pattern))
    consolidated_output = {}
    print(f"Files for aggregation:")
    for f in all_files:
        print(f"\t{f}")
    for file_path in all_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                consolidated_output.update(data)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            continue

    try:
        with open(final_output, "w") as f:
            json.dump(consolidated_output, f, default=default)
    except Exception as e:
        logging.error(f"Error saving consolidated file {final_output}: {e}")


if is_distributed:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    rank = 0
    world_size = 1

try:

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    path_results_subs = os.path.join(path_results, folder)
    pathlib.Path(path_results).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_results_subs).mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime("%d%m%Y")

    try:
        print("Using device:", torch.cuda.get_device_name(0))
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    except:
        pass

    xai_rounds = [xai_round]

    # Run benchmarking.
    if __name__ == "__main__":
        start = time.time()

        meta = DATASETS[dataset_name]  # Assume dataset_name is passed as an argument.

        for xai_round in xai_rounds:
            start_round = time.time()
            for i in range(len(meta["indices"])):

                print(f"Starting round {xai_round} {i}...")
                torch.cuda.empty_cache()
                start_idx, end_idx = meta["indices"][i][0], meta["indices"][i][1]
                model_name = meta["model_name"]

                # Get input, outputs settings.
                SETTINGS = setup_tasks(
                    dataset_name=dataset_name, path_assets=path_assets, device=device
                )
                dataset_settings = {dataset_name: SETTINGS[dataset_name]}
                estimator_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

                # Get analyser suite.
                analyser_suite = setup_test_suite_gef(
                    std_adversary=estimator_kwargs["std_adversary"]
                )

                # Get model.
                model = (
                    dataset_settings[dataset_name]["models"][model_name].eval()
                    # .to(device)
                )

                # Reduce the number of samples.
                dataset_settings[dataset_name]["x_batch"] = dataset_settings[
                    dataset_name
                ]["x_batch"][start_idx:end_idx]
                dataset_settings[dataset_name]["y_batch"] = dataset_settings[
                    dataset_name
                ]["y_batch"][start_idx:end_idx]
                dataset_settings[dataset_name]["s_batch"] = dataset_settings[
                    dataset_name
                ]["s_batch"][start_idx:end_idx]

                # Update model-specific xai parameters.
                xai_methods = XAI_METHODS_ALL[xai_round]
                xai_methods_with_kwargs = {
                    **setup_xai_methods_zennit(
                        xai_methods=xai_methods, model=model  # .cpu()
                    ),
                    **setup_xai_methods_captum(
                        xai_methods=xai_methods,
                        x_batch=dataset_settings[dataset_name]["x_batch"][
                            start_idx:end_idx
                        ],
                        gc_layer=dataset_settings[dataset_name]["gc_layers"][
                            model_name
                        ],
                        img_size=estimator_kwargs["img_size"],
                        nr_channels=estimator_kwargs["nr_channels"],
                        nr_segments=50,
                    ),
                }
                print(f"XAI methods: {xai_methods_with_kwargs.keys()}")

                # Load metrics.
                estimators_meta = setup_metrics_meta(
                    features=estimator_kwargs["features"],
                    patch_size=estimator_kwargs["patch_size"],
                    num_classes=estimator_kwargs["num_classes"],
                    layer_indices=estimator_kwargs["layer_indices"],
                    layer_names=estimator_kwargs["layer_names"],
                    **gef_kwargs,
                )

                ###########################
                # Benchmarking settings. #
                ###########################

                # Define master!
                master = MetaEvaluation(
                    test_suite=analyser_suite,
                    xai_methods=xai_methods_with_kwargs,
                    iterations=3,
                    fname=fname,
                    nr_perturbations=5,
                )

                # Benchmark!
                benchmark = MetaEvaluationBenchmarking(
                    master=master,
                    estimators=estimators_meta,
                    experimental_settings=dataset_settings,
                    path=path_results,
                    folder=folder,
                    write_to_file=True,
                    keep_results=True,
                    channel_first=True,
                    softmax=True,
                    save=True,
                    batch_size=len(dataset_settings[dataset_name]["x_batch"]),
                    device=device,
                )()

                torch.cuda.empty_cache()
                print(f"Finished one round... {time.time() - start_round} sec.")

            # Run consolidation in the main process
            if dist.get_rank() == 0:
                print(today, xai_methods_with_kwargs, dataset_name, xai_round, fname)
                consolidate_meta_evaluation_outputs(
                    base_path=path_results_subs,
                    output_pattern=f"*{today}*{dataset_name}*{fname}",
                    final_output=f"{path_results}{today}_meta_evaluation_consolidated_{dataset_name}_{meta['model_name']}_{xai_round}_{fname}",
                )

    print(f"===========================================")
    print(
        f"Finishing meta-evalaution experiment done in {(time.time() - start):.2f} sec."
    )
    print(f"===========================================")

except Exception as e:
    print(f"An error occurred: {e}")
    # Cleanup or handle the error gracefully.
    dist.destroy_process_group()
    raise e


# Ensure the cleanup is conditional.
if is_distributed and torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

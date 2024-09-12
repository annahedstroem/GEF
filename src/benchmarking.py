"""This module contains the contents for the comparison run."""

from __init__ import PATH_ASSETS, PATH_RESULTS, PATH_RESULTS_IDS

import torch
import os

from datetime import datetime
import time
import pickle
import logging
import numpy as np
import argparse
import torch
import pathlib
import torch.distributed as dist

from settings import (
    EXPLAIN_FUNC,
    XAI_METHODS_MAPPING,
    METRICS,
    METRICS_RANDOM,
    METRICS_DISCOVERY,
    METRICS_LAYER,
    SAVE,
    VERBOSE,
)
from setup_experiments import Experiment
from setup_explanations import get_parameterised_explanations
from common import consolidate_outputs
from src.gef import GEF


# Check if running in a distributed setting.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # update if necessary
is_distributed = "WORLD_SIZE" in os.environ

# import socket
# def find_free_port():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(("", 0))
#         return s.getsockname()[1]


if is_distributed:
    # port = find_free_port()
    # init_method = f"tcp://localhost:{port}"
    # port = os.getenv("MY_DIST_PORT", "29500")  # Default to 29500 if not set
    # init_method = f"tcp://localhost:{port}"
    # print(f"Attempting to bind to port {port} for distributed operations")
    dist.init_process_group(backend="nccl")  # , init_method=init_method)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    rank = 0
    world_size = 1

# Set up argument parser.
parser = argparse.ArgumentParser(description="Process the experiment.")
parser.add_argument("--fname", type=str, required=True, help="Base experiment name.")
parser.add_argument(
    "--datasets", type=str, required=True, help="Base experiment datasets."
)
parser.add_argument(
    "--models", type=str, default=None, required=False, help="Base experiment models."
)
parser.add_argument(
    "--full_size",
    type=int,
    default=None,
    required=False,
    help="Base experiment full_size.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    required=False,
    help="Base experiment batch_size.",
)
parser.add_argument(
    "--xai_methods",
    type=str,
    default=None,
    required=True,
    help="Base XAI methods.",
)
parser.add_argument(
    "--Z",
    type=int,
    required=False,
    help="Base value of Z.",
    default=None,
)
parser.add_argument(
    "--M",
    type=int,
    required=False,
    help="Base value of M.",
    default=None,
)
parser.add_argument(
    "--T",
    type=int,
    required=False,
    help="Base value of T.",
    default=None,
)
parser.add_argument(
    "--K",
    type=int,
    required=False,
    help="Base value of K.",
    default=None,
)
parser.add_argument(
    "--top_K",
    type=int,
    required=False,
    help="Base value of top_K.",
    default=None,
)
parser.add_argument(
    "--is_top_K",
    type=bool,
    required=False,
    help="Base value of is_top_K.",
    default=False,
)
parser.add_argument(
    "--am_steps", type=int, required=False, help="Base value of am_steps.", default=10
)
parser.add_argument(
    "--batch_id",
    type=int,
    required=False,
    help="Base value to run on a fixed batch_id.",
    default=-1,
)
parser.add_argument(
    "--nr_batches",
    type=int,
    required=False,
    help="Base value of nr batches to run from batch_id.",
    default=1,
)
parser.add_argument(
    "--start_idx",
    type=int,
    required=False,
    help="Base start index for the values.",
    default=None,
)
parser.add_argument(
    "--only_naive", type=bool, required=False, help="If run only naive.", default=False
)
parser.add_argument(
    "--only_exact", type=bool, required=False, help="If run only exact.", default=False
)
parser.add_argument(
    "--run_random",
    type=bool,
    required=False,
    help="If run random test.",
    default=False,
)
parser.add_argument(
    "--run_discovery",
    type=bool,
    required=False,
    help="If run discovery test.",
    default=False,
)
parser.add_argument(
    "--run_layer",
    type=bool,
    required=False,
    help="If run layer analysis test.",
    default=False,
)

# Parse arguments.
args = parser.parse_args()
fname = f"{datetime.today().strftime('%d%m%Y')}_{args.fname}"
gef_kwargs = {}
if args.Z is not None:
    gef_kwargs["Z"] = int(args.Z)
if args.M is not None:
    gef_kwargs["M"] = int(args.M)
if args.T is not None:
    gef_kwargs["T"] = int(args.T)
if args.K is not None:
    gef_kwargs["K"] = int(args.K)

# Task kwargs.
dataset_names = args.datasets.split(",")
model_names = (
    args.models.split(",")
    if args.models is not None
    else [None for _ in range(len(dataset_names))]
)
xai_methods = [XAI_METHODS_MAPPING[xai] for xai in args.xai_methods.split(",")]

# Dataset kwargs.
full_size = int(args.full_size) if args.full_size is not None else None
batch_size = int(args.batch_size) if args.batch_size is not None else None
start_idx = int(args.start_idx) if args.start_idx is not None else None
batch_id_fixed = int(args.batch_id) if args.batch_id is not None else -1
nr_batches = int(args.nr_batches) if args.nr_batches is not None else 1

# Hyperparam kwargs.
top_K = int(args.top_K) if args.top_K is not None else 5
is_top_K = args.is_top_K if args.is_top_K is not None else False
am_steps = int(args.am_steps) if args.am_steps is not None else 10

# Metric kwargs.
only_naive = args.only_naive if args.only_naive is not None else False
only_exact = args.only_exact if args.only_exact is not None else False
run_random = args.run_random if args.run_random is not None else False
run_discovery = args.run_discovery if args.run_discovery is not None else False
run_layer = args.run_layer if args.run_layer is not None else False

if batch_id_fixed != -1:
    fname = f"{fname}_batch{batch_id_fixed}"
    if int(nr_batches - 1) != 1:
        fname += f"_to{int(batch_id_fixed + (nr_batches - 1))}"

# Paths based on fname.
PATH_ALL = os.path.join(PATH_RESULTS, fname)
PATH_RESULTS_SUBS = os.path.join(PATH_ALL, "subs")
PATH_RESULTS_PLOT = os.path.join(PATH_ALL, "plots")
pathlib.Path(PATH_ALL).mkdir(parents=True, exist_ok=True)
pathlib.Path(PATH_RESULTS_SUBS).mkdir(parents=True, exist_ok=True)
pathlib.Path(PATH_RESULTS_PLOT).mkdir(parents=True, exist_ok=True)
PATH_RESULTS_SCORES = os.path.join(PATH_RESULTS_SUBS, f"scores_r{rank}_{fname}.pkl")
PATH_RESULTS_SETUPS = os.path.join(PATH_RESULTS_SUBS, f"setups_r{rank}_{fname}.pkl")
PATH_RESULTS_LOGGING = os.path.join(PATH_ALL, f"logging_{fname}.log")

try:

    print(f"Current rank: {dist.get_rank()}.")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    try:
        print("Using device:", torch.cuda.get_device_name(0))
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
    except:
        pass

    scores_dict = {}
    setups = {}

    print(f"===========================================")
    print(f"Running comparison experiment {fname}...")
    print(f"===========================================")
    print("Verbose mode:", VERBOSE)

    for dataset_name, model_name in zip(dataset_names, model_names):

        # Fetch the experimental details.
        experiment = Experiment(
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
            full_size=full_size,
            batch_size=batch_size,
        )
        model, model_name = experiment.model, experiment.model_name
        model.eval()
        model.to(device)

        setting_name = f"({dataset_name}, {model_name})"
        print(f"\nGenerating {experiment.config.task} task - {setting_name}....")

        scores_dict[setting_name] = {}

        if run_random:
            METRICS = METRICS_RANDOM
        if run_discovery:
            METRICS = METRICS_DISCOVERY
        if run_layer:
            METRICS = METRICS_LAYER

        for metric_name, metric_init in METRICS.items():

            # Check if exact_mode is set to True of metric_init.
            if only_naive and metric_init.exact_mode is True:
                continue
            if only_exact and metric_init.exact_mode is False:
                continue

            if metric_init.name == "GEF":
                metric_init.num_classes = experiment.config.num_classes

            for key, value in gef_kwargs.items():
                setattr(metric_init, key, value)

            if experiment.config.task == "text" and "input" in metric_name:
                # TODO. Extend input noise function so that type is in indices (dtype: long).
                print(
                    f"Skipping {metric_name} for text task as perturbation function for text (dtype: long) is missing."
                )
                continue

            # Update model-specific xai parameters.
            print(f"All XAI methods {xai_methods}\n")
            xai_methods_with_kwargs = get_parameterised_explanations(
                task=experiment.config.task,
                xai_methods=xai_methods,
                device=device,
                model=model.cuda(),  # TODO. This might be causing a problem model.cuda(),
                am_batch=None,
                llm_explainer_name="google/gemma-2b-it",  # meta-llama/Meta-Llama-3-8B
                top_K=top_K,
                is_top_K=is_top_K,
                tokenizer=experiment.tokenizer,
                am_steps=am_steps,
                nr_channels=experiment.config.nr_channels,
                xai_layer_name=experiment.xai_layer_name,
                img_size=experiment.config.img_size,
                class_labels=experiment.config.class_labels,
                subtask=experiment.config.feature_col_name,
                tokenizer_max_length=experiment.config.token_max_length,
            )

            if VERBOSE:
                print(f"\n\t{metric_name}\n{xai_methods_with_kwargs.keys()}")
            scores_dict[setting_name][metric_name] = {}

            for xai_method, explain_func_kwargs in xai_methods_with_kwargs.items():
                print(f"\n\t\t{xai_method}")

                # Initialise the scores dictionary for the batch.
                scores_dict_batch = {
                    "scores": np.array([]),
                    "scores_per_model": np.array([]),
                    "distortion_f": np.array([]),
                    "distortion_e": np.array([]),
                    "perturbation_path": np.array([]),
                    "time": 0,
                    "nr_samples": 0,
                    "accuracy": 0,
                    # "extras": {},
                    "model_distortions_by_layer": dict(),
                    "explanation_distortions_by_layer": dict(),
                }

                # Update random top K if several are passed.
                if "K=" in xai_method:
                    explain_func_kwargs["top_K"] = int(xai_method.split("K=")[1])

                # Print all parameters of the metric init.
                # if VERBOSE:
                # print(list(metric_init.__dict__.items()))

                if "top_K" in explain_func_kwargs:
                    print(f"top_K={explain_func_kwargs['top_K']}")

                for batch_id, (x_batch, y_batch, am_batch) in enumerate(
                    experiment.generate_batch()
                ):

                    # To start with later samples.
                    if start_idx is not None:
                        curr_batch_id = int(batch_id + 1)
                        curr_index = int(len(x_batch) * curr_batch_id)
                        if start_idx < curr_index:
                            continue

                    if VERBOSE:
                        index_random = np.random.randint(0, x_batch.shape[0])
                        print(
                            f"\n\t\t\tshapes: {x_batch.shape}, {y_batch.shape}, dtypes: {x_batch.dtype}, {y_batch.dtype}, "
                            f"{am_batch.shape if am_batch is not None else None}, "
                            # f"{x_batch[0][0] if experiment.config.task == 'text' else None} "
                            # f"{y_batch[0] if experiment.config.task == 'text' else None} "
                            # f"\n{experiment.tokenizer.convert_ids_to_tokens(x_batch[index_random]) if experiment.config.task == 'text' else None}"
                        )

                    if am_batch is not None:
                        for xai in xai_methods_with_kwargs:
                            if xai_methods_with_kwargs[xai]["task"] == "text":
                                xai_methods_with_kwargs[xai][
                                    "attention_mask"
                                ] = am_batch

                    # Initialise and run the GEF metric.
                    if metric_init.name == "GEF":

                        if "Auto" in xai_method:
                            # Set normalise to False.
                            metric_init.normalise = False
                        else:
                            metric_init.normalise = True

                    start = time.time()
                    if metric_init.name == "GEF":
                        scores = metric_init(
                            model=model,
                            x_batch=x_batch,
                            y_batch=y_batch,
                            a_batch=None,
                            custom_batch=am_batch,
                            device=device,
                            batch_size=len(
                                x_batch
                            ),  # experiment.config.batch_size,  # Update this if necessary.
                            explain_func=EXPLAIN_FUNC,
                            explain_func_kwargs=explain_func_kwargs,
                        )

                        if scores_dict_batch["scores"].size == 0:

                            # Update the batch dictionary.
                            scores_dict_batch["scores"] = np.array(scores)
                            scores_dict_batch["accuracy"] = metric_init.accuracy
                            scores_dict_batch["scores_per_model"] = metric_init.scores
                            scores_dict_batch["distortion_f"] = metric_init.distortion_f
                            scores_dict_batch["distortion_e"] = metric_init.distortion_e
                            scores_dict_batch["perturbation_path"] = (
                                metric_init.perturbation_path
                            )
                        else:
                            scores_dict_batch["scores"] = np.append(
                                scores_dict_batch["scores"],
                                scores,  # gef_scores
                            )
                            scores_dict_batch["scores_per_model"] = np.append(
                                scores_dict_batch["scores_per_model"],
                                metric_init.scores,  # M, scores
                            )
                            scores_dict_batch["accuracy"] = np.append(
                                scores_dict_batch["accuracy"],
                                metric_init.accuracy,
                            )
                            scores_dict_batch["distortion_f"] = np.concatenate(
                                (
                                    scores_dict_batch["distortion_f"],
                                    metric_init.distortion_f,
                                ),
                                axis=-1,
                            )
                            scores_dict_batch["distortion_e"] = np.concatenate(
                                (
                                    scores_dict_batch["distortion_e"],
                                    metric_init.distortion_e,
                                ),
                                axis=-1,
                            )
                            scores_dict_batch["perturbation_path"] = np.append(
                                scores_dict_batch["perturbation_path"],
                                metric_init.perturbation_path,
                            )

                    else:

                        x_batch = x_batch.detach().cpu().numpy()
                        y_batch = y_batch.detach().cpu().numpy()
                        model = model.cpu()
                        scores = metric_init(
                            model=model,
                            x_batch=x_batch,
                            y_batch=y_batch,
                            a_batch=None,
                            device=device,
                            batch_size=len(
                                x_batch
                            ),  # experiment.config.batch_size,  # Update this if necessary.
                            explain_func=EXPLAIN_FUNC,
                            explain_func_kwargs=explain_func_kwargs,
                        )

                        # Add extras from the layer analysis.
                        if "Layer" in metric_init.name:

                            # Check if scores_dict_batch["model_distortions_by_layer"] is an emtpy dict.
                            if not bool(
                                scores_dict_batch["model_distortions_by_layer"]
                            ):
                                scores_dict_batch["model_distortions_by_layer"] = (
                                    metric_init.model_distortions_by_layer
                                )
                                scores_dict_batch[
                                    "explanation_distortions_by_layer"
                                ] = metric_init.explanation_distortions_by_layer
                            else:
                                # Append the results to the actual scores values  Dict[str, List[float]].
                                for (
                                    key
                                ) in metric_init.model_distortions_by_layer.keys():
                                    scores_dict_batch["model_distortions_by_layer"][
                                        key
                                    ] = np.append(
                                        scores_dict_batch["model_distortions_by_layer"][
                                            key
                                        ],
                                        metric_init.model_distortions_by_layer[key],
                                    )
                                    scores_dict_batch[
                                        "explanation_distortions_by_layer"
                                    ][key] = np.append(
                                        scores_dict_batch[
                                            "explanation_distortions_by_layer"
                                        ][key],
                                        metric_init.explanation_distortions_by_layer[
                                            key
                                        ],
                                    )

                        if scores_dict_batch["scores"].size == 0:

                            # Update the batch dictionary.
                            scores_dict_batch["scores"] = np.array(scores)
                        else:
                            scores_dict_batch["scores"] = np.append(
                                scores_dict_batch["scores"],
                                scores,
                            )

                    end = time.time()

                    scores_dict_batch["time"] += end - start
                    scores_dict_batch["nr_samples"] += len(x_batch)

                    if VERBOSE:
                        print(f"\t\t\ttime {end - start}")
                        print(
                            f"\t\t\t{setting_name} x {metric_name} x {xai_method} - scores: {np.nanmean(scores):.3f}, (±{np.nanstd(scores):.3f}) -\n{np.round(np.array(scores), 2)}"
                        )

                    del scores

                    if metric_init.name == "GEF":
                        setups[metric_name] = (
                            f"N:{experiment.config.batch_size}, M:{metric_init.M}, Z:{metric_init.Z}"
                            + (
                                f", T:{metric_init.T}, K:{metric_init.K}"
                                if getattr(metric_init, "T", None) is not None
                                else ""
                            )
                        )
                    torch.cuda.empty_cache()

                    if batch_id_fixed != -1:
                        if int(batch_id_fixed + (nr_batches - 1)) == batch_id:
                            break

                # Append the results to the actual scores dict.
                scores_dict[setting_name][metric_name][xai_method] = scores_dict_batch

    if SAVE:

        # Configure logging.
        # logging.basicConfig(
        #    filename=PATH_RESULTS_LOGGING,
        #    level=logging.INFO,
        #    format="%(asctime)s - %(message)s",
        # )
        # log_file_content(file_path="comparison_settings.py")

        # Save scores_dict and setups.
        with open(f"{PATH_RESULTS_SCORES}", "wb") as f:
            pickle.dump(scores_dict, f)
        with open(f"{PATH_RESULTS_SETUPS}", "wb") as f:
            pickle.dump(setups, f)

        print(f"{PATH_RESULTS_SCORES} saved at {fname}.")

    print(f".... Consolidating results for {fname} ....")

    # Run consolidation in the main process
    if dist.get_rank() == 0:
        consolidate_outputs(
            PATH_RESULTS_SUBS,
            "*setups_r*.pkl",
            f"{PATH_ALL}/setups_consolidated_{fname}.pkl",
        )
        consolidate_outputs(
            PATH_RESULTS_SUBS,
            "*scores_r*.pkl",
            f"{PATH_ALL}/scores_consolidated_{fname}.pkl",
        )

    print(f"===========================================")
    print(f"Finishing comparison experiment {fname} done.")
    print(f"===========================================")

except Exception as e:
    print(f"An error occurred: {e}")
    # Cleanup or handle the error gracefully.
    dist.destroy_process_group()
    raise e


# Ensure the cleanup is conditional.
if is_distributed and torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

"""This module contains raw scores post-processing for plotting."""

import pandas as pd
import numpy as np
import pickle
from typing import Optional
from scipy.stats import spearmanr
from src.helpers.experiments.params import META, REPLACE, GLOBAL_METHODS


def convert_dict_to_df(
    filename,
    method_suffix_name: Optional[str] = None,
    method_suffix: Optional[str] = None,
    keep_only_first_model: bool = False,
    remove_first_model: bool = False,
    verbose: bool = False,
):
    with open(filename, "rb") as f:
        scores_dict = pickle.load(f)

    # Initialise dictionaries to store the results.
    results = {
        "Task": [],
        "Dataset": [],
        "Model": [],
        "mean_score": [],
        "std_score": [],
        "std_error_score": [],
        "time": [],
        "model_size": [],
        "explanation_size": [],
        "mean_e_dist": [],
        "std_e_dist": [],
        "std_error_e_dist": [],
        "mean_f_dist": [],
        "std_f_dist": [],
        "std_error_f_dist": [],
        "gef_scores": [],
        "model_distortions": [],
        "explanation_distortions": [],
        "perturbation_path": [],
        "accuracy": [],
        "sample_size": [],
        "rank": [],
    }
    indices = []

    # Loop through the scores_dict to extract and compute the necessary metrics.
    for setting_name, data in scores_dict.items():
        for metric_name, score_data in data.items():
            mean_scores = []
            for xai_method, metrics in score_data.items():

                if (method_suffix and xai_method == method_suffix_name) or (
                    method_suffix and method_suffix_name is None
                ):
                    xai_method += method_suffix

                distortion_f = np.array(metrics["distortion_f"])
                distortion_e = np.array(metrics["distortion_e"])

                # Recalculate scores using calculate_similarity_scores_row.

                try:
                    scores = recalculate_similarity_scores_row(
                        model_distortions=distortion_f,
                        explanation_distortions=distortion_e,
                        keep_only_first_model=keep_only_first_model,
                        remove_first_model=remove_first_model,
                    )
                except:
                    # if verbose:
                    # print(
                    #    "fFailed to recalculate_similarity_scores_row for {setting_name} {metric_name} {xai_method}"
                    # )
                    scores = np.array(metrics["scores"])

                time = metrics["time"]
                sample_size = metrics["nr_samples"]

                if verbose:
                    print(
                        f"Processing {setting_name}, {metric_name}, {xai_method}, nr scores: {len(scores)}"
                    )
                mean_score = np.nanmean(scores)
                std_score = np.nanstd(scores)
                perturbation_path = metrics["perturbation_path"]
                accuracy = metrics.get("accuracy", None)

                mean_distortion_f = np.nanmean(distortion_f)
                std_distortion_f = np.nanstd(distortion_f)
                mean_distortion_e = np.nanmean(distortion_e)
                std_distortion_e = np.nanstd(distortion_e)

                std_error_score = np.nanstd(scores) / np.sqrt(sample_size)
                std_error_distortion_e = np.nanstd(distortion_e) / np.sqrt(sample_size)
                std_error_distortion_f = np.nanstd(distortion_f) / np.sqrt(sample_size)

                # Append the computed metrics to the results.
                mean_scores.append(mean_score)

                results["mean_score"].append(mean_score)
                results["std_score"].append(std_score)
                results["std_error_score"].append(std_error_score)
                results["mean_f_dist"].append(mean_distortion_f)
                results["std_f_dist"].append(std_distortion_f)
                results["std_error_f_dist"].append(std_error_distortion_e)
                results["mean_e_dist"].append(mean_distortion_e)
                results["std_e_dist"].append(std_distortion_e)
                results["std_error_e_dist"].append(std_error_distortion_f)
                results["time"].append(time)
                results["perturbation_path"].append(perturbation_path)
                results["accuracy"].append(accuracy)
                results["sample_size"].append(sample_size)

                results["gef_scores"].append(scores.tolist())
                results["model_distortions"].append(distortion_f.tolist())
                results["explanation_distortions"].append(distortion_e.tolist())

                results["Task"].append(META[setting_name]["task"])
                results["model_size"].append(META[setting_name]["model_size"])
                results["explanation_size"].append(
                    META[setting_name]["explanation_size"]
                )
                results["Dataset"].append(setting_name.split(", ")[0].replace("(", ""))
                results["Model"].append(setting_name.split(", ")[1].replace(")", ""))

                # Append the corresponding index.
                indices.append(
                    (
                        REPLACE.get(setting_name, setting_name),
                        REPLACE.get(metric_name, metric_name),
                        REPLACE.get(xai_method, xai_method),
                    )
                )

            # Compute the rank for the mean scores (higher is better).
            ranks = pd.Series(mean_scores).rank(ascending=False)
            for rank in ranks:
                results["rank"].append(rank)

    # Create a MultiIndex DataFrame.
    indices = pd.MultiIndex.from_tuples(
        indices, names=["Setting", "Metric", "XAI Method"]
    )
    df = pd.DataFrame(results, index=indices)
    df.reset_index(inplace=True)

    df["Scope"] = df["XAI Method"].apply(
        lambda x: "Global" if x in GLOBAL_METHODS else "Local"
    )
    df.loc[df["XAI Method"] == "RAN", "Scope"] = "RAN"

    return df


def convert_layer_dict_to_df(
    filename,
    method_suffix_name: Optional[str] = None,
    method_suffix: Optional[str] = None,
):

    with open(filename, "rb") as f:
        scores_dict = pickle.load(f)

    # Initialise dictionaries to store the results.
    results = {
        "Task": [],
        "Dataset": [],
        "Model": [],
        "layers": [],
        "model_distortions_by_layer": [],
        "explanation_distortions_by_layer": [],
        "model_distortions": [],
        "explanation_distortions": [],
        "time": [],
        "model_size": [],
        "explanation_size": [],
        "model_distortions": [],
        "explanation_distortions": [],
        "sample_size": [],
    }
    indices = []

    # Loop through the scores_dict to extract and compute the necessary metrics
    for setting_name, data in scores_dict.items():
        for metric_name, score_data in data.items():
            mean_scores = []
            for xai_method, metrics in score_data.items():

                if (method_suffix and xai_method == method_suffix_name) or (
                    method_suffix and method_suffix_name is None
                ):
                    xai_method += method_suffix

                layers = list(metrics["model_distortions_by_layer"].keys())
                model_distortions_by_layer = metrics["model_distortions_by_layer"]
                explanation_distortions_by_layer = metrics[
                    "explanation_distortions_by_layer"
                ]
                model_distortions_values = list(
                    metrics["model_distortions_by_layer"].values()
                )
                explanation_distortions_values = list(
                    metrics["explanation_distortions_by_layer"].values()
                )

                # Recalculate scores using recalculate_similarity_scores_row.
                time = metrics["time"]
                sample_size = metrics["nr_samples"]

                results["layers"].append(layers)
                results["model_distortions_by_layer"].append(model_distortions_by_layer)
                results["explanation_distortions_by_layer"].append(
                    explanation_distortions_by_layer
                )
                results["model_distortions"].append(model_distortions_values)
                results["explanation_distortions"].append(
                    explanation_distortions_values
                )
                results["time"].append(time)
                results["sample_size"].append(sample_size)

                results["Task"].append(META[setting_name]["task"])
                results["model_size"].append(META[setting_name]["model_size"])
                results["explanation_size"].append(
                    META[setting_name]["explanation_size"]
                )
                results["Dataset"].append(setting_name.split(", ")[0].replace("(", ""))
                results["Model"].append(setting_name.split(", ")[1].replace(")", ""))

                # Append the corresponding index.
                indices.append(
                    (
                        REPLACE.get(setting_name, setting_name),
                        REPLACE.get(metric_name, metric_name),
                        REPLACE.get(xai_method, xai_method),
                    )
                )

    # Create a MultiIndex DataFrame.
    indices = pd.MultiIndex.from_tuples(
        indices, names=["Setting", "Metric", "XAI Method"]
    )
    df = pd.DataFrame(results, index=indices)
    df.reset_index(inplace=True)

    df["Metric"] = df["Metric"].str.replace("Bottom", "Bottom-up")
    df["Metric"] = df["Metric"].str.replace("Top", "Top-down")

    df["Scope"] = df["XAI Method"].apply(
        lambda x: "Global" if x in GLOBAL_METHODS else "Local"
    )
    df.loc[df["XAI Method"] == "RAN", "Scope"] = "RAN"

    return df


def recalculate_similarity_scores_row(
    row: Optional[pd.Series] = None,
    model_distortions: Optional[np.ndarray] = None,
    explanation_distortions: Optional[np.ndarray] = None,
    similarity_func: Optional[callable] = spearmanr,
    keep_only_first_model: bool = False,
    remove_first_model: bool = False,
) -> float:
    """
    Calculate similarity scores between distortions using the specified similarity function for a single row.

    Parameters
    ----------
    row (pandas.Series):
        A single row from a DataFrame containing model and explanation distortions.
    model_distortions (numpy.ndarray):
        Distortions of the model.
    explanation_distortions (numpy.ndarray):
        Distortions of the explanation.
    similarity_func (callable):
        Function to calculate similarity between two vectors (default is spearmanr).
    keep_only_first_model (bool):
        Whether to keep only the first model (default is False).
    remove_first_model (bool):
        Whether to remove the first model (default is False).

    """
    if row is not None:
        distortion_f = np.array(row["model_distortions"])
        distortion_e = np.array(row["explanation_distortions"])
    else:
        distortion_f = np.array(model_distortions)
        distortion_e = np.array(explanation_distortions)

    M, N, batch_size = distortion_f.shape
    scores = np.zeros((M, batch_size)) * np.nan

    for m_ix in range(M):
        if keep_only_first_model and m_ix > 0:
            continue
        elif remove_first_model and m_ix == 0:
            continue
        for s_ix in range(batch_size):
            if not np.isnan(distortion_e[m_ix, :, s_ix]).any():
                sim_score = similarity_func(
                    distortion_f[m_ix, :, s_ix], distortion_e[m_ix, :, s_ix]
                )[0]
                scores[m_ix, s_ix] = sim_score
    if remove_first_model:
        # Remove first dim of scores.
        scores = scores[1:, :]
    if keep_only_first_model:
        # Keep only first dim of scores.
        scores = scores[:1, :]

    gef_scores = np.nanmean(scores, axis=0)
    return gef_scores

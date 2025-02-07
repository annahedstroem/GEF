"""This module contains postprocessing and plotting code."""

import random
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

from metaquantus import make_benchmarking_df

from src.helpers.configs import *
from .postprocess import *
from .plotting_faithfulness import *


def plot_agreement_cm(df, cmap: str = "Greens", dtype: str = "tab", order: bool = True):

    df.sort_values(by="xai method", ascending=order, inplace=True)

    # Create a pivot table for the confusion matrix.
    pivot_df = df.pivot(
        index="xai method", columns="Metric", values=["mean_score", "rank"]
    )

    # Combine mean_score and rank into a single cell for annotation.
    pivot_df_combined = pivot_df.apply(
        lambda row: [
            f'{row["mean_score"][col]:.2f}\n(R{int(row["rank"][col])})'
            for col in row["mean_score"].index
        ],
        axis=1,
        result_type="expand",
    )
    pivot_df_combined.columns = pivot_df["mean_score"].columns

    # Plot the heatmap with only the mean_score values.
    plt.figure(figsize=(3, 2.5))
    heatmap = sns.heatmap(
        pivot_df["mean_score"].T,
        annot=False,
        cmap=cmap,
        linewidths=0.5,
        linecolor="black",
        cbar=True,
        vmin=0,
        vmax=1,
    )
    for i in range(pivot_df_combined.shape[0]):
        for j in range(pivot_df_combined.shape[1]):
            c = "black"
            # If value is over 0.75, make colour white.
            if float(pivot_df_combined.iloc[i, j].split("\n")[0]) >= 0.75:
                c = "white"
            heatmap.text(
                i + 0.5,
                j + 0.5,
                pivot_df_combined.iloc[i, j],
                ha="center",
                va="center",
                color=c,
            )
    plt.xlabel(None)
    plt.ylabel(None)
    plt.savefig(f"plots/ranking_agreement_heatmap_{dtype}.svg")
    plt.show()


def plot_agreement(
    df, dtype: str = "tab", ncols: int = 3, colors: dict = {}, order: bool = True
):

    df.sort_values(by="xai method", ascending=order, inplace=True)
    plt.figure(figsize=(3, 3))
    DIFF = 0.25

    # Plot lines for each XAI method.
    for method in df["xai method"].unique():
        method_data = df[df["xai method"] == method]
        plt.plot(
            method_data["Metric"],
            method_data["mean_score"],
            marker="o",
            linestyle="-",
            linewidth=6,
            path_effects=[pe.Stroke(linewidth=5, foreground="black"), pe.Normal()],
            alpha=0.9,
            label=method,
            color=colors[method],
        )

        for x, y, rank in zip(
            method_data["Metric"], method_data["mean_score"], method_data["rank"]
        ):
            plt.text(
                x,
                y + 0.05,
                f"(R{int(rank)})",
                color=colors[method],
                ha="center",
                va="bottom",
            )

    plt.ylabel("GEF Score")
    plt.ylim(df["mean_score"].min() - DIFF, df["mean_score"].max() + DIFF)
    plt.xlim(-0.25, 1.25)
    plt.title(df.Setting[0].replace("(", "").replace(", ", " ("))
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", ncols=ncols
    )  # title='Methods',
    plt.grid(True)
    plt.savefig(f"plots/agreement_{dtype}.svg")
    plt.show()


def plot_faithfulness_curve_patches(
    confidences,
    patch_sizes,
    colour_map,
    linestyles,
    dataset,
    img_size: int = 28,
    title: Optional[str] = None,
    distortion: bool = True,
):
    total_pixels = img_size * img_size
    x_values = np.arange(1, total_pixels + 1) / total_pixels * 100

    start_confidence = confidences[0][0]

    for i, patch_size in enumerate(patch_sizes):
        if distortion:
            confidences[i] = np.abs(np.subtract(confidences[i], start_confidence))
        plt.plot(
            x_values,
            confidences[i],
            color=colour_map[patch_size],
            label=f"{patch_size}x{patch_size} patch",
            linestyle=linestyles[dataset],
            lw=1.5,
        )

    end_confidence = 0
    if distortion:
        end_confidence = -start_confidence
        start_confidence = 0
    linear_confidences = np.linspace(start_confidence, end_confidence, total_pixels)
    plt.plot(
        x_values,
        np.abs(linear_confidences),
        linestyle=linestyles[dataset],
        c="slategrey",
        alpha=0.5,
        label="Expectation",
    )

    plt.xlabel("Cumulative Perturbation (%)")
    if distortion:
        plt.ylabel("Model Distortion")
    else:
        plt.ylabel("Model Response")
    if title:
        plt.title(title)
    if dataset == "MNIST":
        plt.legend(title="Input Parameter", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Model Faithfulness (Ass. 3)")


def plot_faithfulness_curve_baselines(
    confidences,
    modifications,
    colour_map,
    linestyles,
    dataset,
    img_size: int = 28,
    title: Optional[str] = None,
    distortion: bool = False,
    ADD_DATASET_LABEL: bool = False,
):
    total_pixels = img_size * img_size
    x_values = np.arange(1, total_pixels + 1) / total_pixels * 100

    start_confidence = confidences[0][0][0]

    for m, modification in enumerate(modifications):
        sample_confidences = confidences[m][0]
        if distortion:
            sample_confidences = np.abs(
                np.subtract(sample_confidences, start_confidence)
            )
        plt.plot(
            x_values,
            sample_confidences,
            color=colour_map[modification],
            lw=1.5,
            linestyle=linestyles[dataset],
            label=modification.title(),
        )

    end_confidence = 0
    if distortion:
        end_confidence = -start_confidence
        start_confidence = 0
    linear_confidences = np.linspace(start_confidence, end_confidence, total_pixels)
    plt.plot(
        x_values,
        np.abs(linear_confidences),
        linestyle=linestyles[dataset],
        c="slategrey",
        alpha=0.5,
        label="Expectation",
    )

    plt.xlabel("Cumulative Perturbation (%)")
    if distortion:
        plt.ylabel("Model Distortion")
    else:
        plt.ylabel("Model Response")
    if title:
        plt.title(title)
    if dataset == "MNIST":
        plt.legend(
            title="Input Parameter", loc="center left", bbox_to_anchor=(1, 0.5)
        )  # plt.legend()

    datasets = ["MNIST", "fMNIST"]
    plt.title("Model Faithfulness (Ass. 3)")
    plt.grid(True)

    if ADD_DATASET_LABEL:

        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                hatch="." if key == "fMNIST" else "",
                edgecolor="black",
            )
            for key in datasets
        ]
        labels = datasets
        plt.legend(
            handles,
            labels,
            title="Datasets",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )


def plot_alignment(df_robustness, noise_types, noise_levels):

    colours_discovery = {
        "(COMPAS, 3-layer MLP)": "#81D46D",
        "(Avila, 2-layer MLP)": "#D8AD3A",
        "(ImageNet, ResNet18)": "#70A1CE",
        "(Path, MedCNN)": "lightcoral",
        "(MNIST, LeNet)": "sandybrown",
    }

    for noise_type in noise_types:
        df_plot = df_robustness[df_robustness["Metric"].str.contains(noise_type)]

        M = 5
        Z = 10
        nr_samples = df_plot.sample_size.iloc[0]
        PLOT_CONTOUR = True

        # Go over all the settings.
        settings_disc = df_plot.Setting.unique()
        perturbation_levels = np.arange(0, Z)

        for s in settings_disc:

            colour = colours_discovery[s]
            df_robustness_setting = df_plot.loc[df_plot.Setting == s]

            xai_methods = df_robustness_setting["xai method"].unique()

            for ix, xai in enumerate(xai_methods):
                df_robustness_setting_xai = df_robustness_setting.loc[
                    df_robustness_setting["xai method"] == xai
                ]
                model_distortions = np.array(
                    df_robustness_setting.model_distortions.iloc[ix]
                ).mean(axis=0)
                explanation_distortions = np.array(
                    df_robustness_setting.explanation_distortions.iloc[ix]
                ).mean(axis=0)

                fig, ax = plt.subplots(figsize=(3, 3))

                for i in range(Z):
                    if i in noise_levels:
                        colour = f"#{random.randint(0, 0xFFFFFF):06X}"  # Get a random hex colour.

                        if PLOT_CONTOUR:
                            xi = model_distortions[i]
                            yi = explanation_distortions[i]
                            X, Y = np.meshgrid(
                                np.linspace(xi.min(), xi.max(), 100),
                                np.linspace(yi.min(), yi.max(), 100),
                            )
                            Z_values = np.sqrt(
                                (X - xi.mean()) ** 2 + (Y - yi.mean()) ** 2
                            )
                            xy = np.vstack(
                                [model_distortions[i], explanation_distortions[i]]
                            )
                            kde = gaussian_kde(xy, bw_method=0.5)

                            X, Y = np.meshgrid(
                                np.linspace(
                                    model_distortions[i].min(),
                                    model_distortions[i].max(),
                                    100,
                                ),
                                np.linspace(
                                    explanation_distortions[i].min(),
                                    explanation_distortions[i].max(),
                                    100,
                                ),
                            )
                            positions = np.vstack([X.ravel(), Y.ravel()])
                            Z_values = np.reshape(kde(positions).T, X.shape)
                            cmap = ListedColormap([colour])
                            cp = ax.contour(
                                X, Y, Z_values, cmap=cmap, levels=4, alpha=0.6
                            )

                        plt.scatter(
                            x=model_distortions[i],
                            y=explanation_distortions[i],
                            c=colour,
                            alpha=0.5,
                            edgecolor="black",
                            label=f"z={i}",
                        )

                plt.legend(loc="upper right")
                plt.xlim(
                    model_distortions.min() * 0.8, model_distortions.max()
                )  # * 1.1)
                plt.ylim(
                    explanation_distortions.min() * 0.9,
                    explanation_distortions.max() * 1.1,
                )
                plt.xlabel("Model Distortion")
                plt.ylabel(f"{xai} Distortion")
                plt.title(s.replace("(", "").replace(", ", " ("))
                plt.grid(True)
                plt.savefig(
                    f'plots/alignment_{xai}_{s.lower().replace("(", "").replace(", ", "_").replace(")", "")}.svg'
                )
                plt.show()


def plot_robustness(df_robustness, noise_types=["Bridge - Input - Additive - Class"]):
    # Hyperparameters.
    colours_discovery = {
        "(COMPAS, 3-layer MLP)": "#81D46D",
        "(Avila, 2-layer MLP)": "#D8AD3A",
        "(ImageNet, ResNet18)": "#70A1CE",
        "(Path, MedCNN)": "lightcoral",
        "(MNIST, LeNet)": "sandybrown",
    }

    for noise_type in noise_types:

        noise_type_str = noise_type.split("Bridge - ")[1].replace(" -", " ")
        df_plot = df_robustness[df_robustness["Metric"].str.contains(noise_type)]

        M = 5
        Z = 10
        nr_samples = df_plot.sample_size.iloc[0]
        PLOT_VIOLIN = True
        PLOT_SCATTER = False
        PLOT_OUTLIERS = False
        PLOT_JITTER = True

        # Go over all the settings.
        settings_disc = df_plot.Setting.unique()
        perturbation_levels = np.arange(0, Z)

        for s in settings_disc:
            colour = colours_discovery[s]
            df_robustness_setting = df_plot.loc[df_plot.Setting == s]
            xai_methods = len(df_robustness_setting)
            model_distortions = (
                np.array(
                    [
                        np.array(df_robustness.model_distortions.iloc[i]).mean(axis=0)
                        for i in range(xai_methods)
                    ]
                )
                .transpose(1, 2, 0)
                .reshape(Z, nr_samples * xai_methods)
            )

            fig, ax = plt.subplots(figsize=(4, 3))

            if PLOT_SCATTER:
                for Df in model_distortions:
                    plt.scatter(x=perturbation_levels, y=Df, alpha=0.25, c=colour)

            if PLOT_VIOLIN:
                data = []
                for i in range(Z):
                    data.append(
                        model_distortions[
                            :, i * nr_samples * M : (i + 1) * nr_samples * M
                        ].flatten()
                    )

                df_points = pd.DataFrame(
                    {
                        "Perturbation": np.repeat(
                            np.arange(Z), nr_samples * xai_methods
                        ),
                        "Model distortion": np.concatenate(data),
                    }
                )

                # Create violin plots for each perturbation level.
                sns.violinplot(
                    x="Perturbation",
                    y="Model distortion",
                    data=df_points,
                    #    inner="point",
                    palette=[colour] * Z,
                    ax=ax,
                )

                if PLOT_JITTER:
                    jitter = 0.025

                    # Calculate the 1.5*IQR for each Perturbation level to determine outliers.
                    for i, level in enumerate(df_points["Perturbation"].unique()):
                        subset = df_points[df_points["Perturbation"] == level]
                        q1 = subset["Model distortion"].quantile(0.05)
                        q3 = subset["Model distortion"].quantile(0.95)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = subset[
                            (subset["Model distortion"] < lower_bound)
                            | (subset["Model distortion"] > upper_bound)
                        ]

                        # Plot all points.
                        ax.scatter(
                            np.random.normal(i, jitter, size=len(subset)),
                            subset["Model distortion"],
                            color=colour,
                            alpha=0.5,
                            edgecolor="black",
                            s=30,
                        )

                        if PLOT_OUTLIERS:
                            # Highlight outliers with a different color and larger size.
                            ax.scatter(
                                np.random.normal(i, jitter, size=len(outliers)),
                                outliers["Model distortion"],
                                color="red",
                                edgecolor="black",
                                s=50,
                                label="Outliers" if i == 0 else "",
                            )

            print(model_distortions.shape)
            plt.errorbar(
                x=perturbation_levels,
                y=model_distortions.mean(axis=1),
                yerr=model_distortions.std(axis=1),
                lw=1,
                color="blue",
                marker="o",
                linestyle="dashed",
                label="Mean",
            )

            noise_levels = np.array(
                [
                    df_robustness_setting.perturbation_path.iloc[i]
                    for i in range(xai_methods)
                ]
            ).mean(axis=0)
            plt.title("Model Robustness (Ass. 1)")
            plt.ylabel("Model Distortion")
            setting_label = s.replace("(", "").replace(", ", " (")
            ax.legend([f"{setting_label}".replace(" (", "\n(")], loc="upper left")
            plt.xlabel("Additive Noise Level ($Z$)")
            plt.grid(True)
            ax.set_xticks(np.arange(Z))
            ax.set_xticklabels(
                ["Acc. Orig"] + np.repeat("", Z - 2).tolist() + ["Acc. = 1/C"]
            )
            plt.savefig(
                f'plots/model_robustness_analysis_{s.lower().replace("(", "").replace(", ", "_").replace(")", "")}.svg'
            )
            plt.show()


def plot_sensitivity(df_sensitivity):

    colours_discovery_dark = {
        "(COMPAS, 3-layer MLP)": "darkgreen",
        "(Avila, 2-layer MLP)": "darkgoldenrod",
        "(ImageNet, ResNet18)": "darkblue",
        "(Path, MedCNN)": "darkred",
        "(MNIST, LeNet)": "darkorange",
    }
    settings_disc = df_sensitivity.Setting.unique()

    for s in settings_disc:

        fig, ax = plt.subplots(figsize=(4, 3))
        df_layers = df_sensitivity.loc[df_sensitivity.Setting == s]
        layers = df_layers.layers.iloc[0]
        ran_orders = ["Top", "Bottom"]

        def concat_lists(series):
            return np.concatenate(series.tolist(), axis=1)

        df_layers_ran = (
            df_layers.groupby("Metric")
            .apply(
                lambda x: pd.Series(
                    {
                        "model_distortions": concat_lists(x["model_distortions"]),
                        "layers": x["layers"].iloc[0],
                    }
                )
            )
            .reset_index()
        )

        for i in range(len(df_layers_ran)):
            dist = df_layers_ran.model_distortions.iloc[i]
            order = df_layers_ran.Metric.iloc[i].split(" - ")[-1]

            if order == "Top":
                # Reverse the x values.
                x_vals = np.arange(len(layers))[::-1]
                colour = colours_discovery[s]
            else:
                # Regular x values.
                x_vals = np.arange(len(layers))
                colour = colours_discovery_dark[s]

            plt.errorbar(
                x_vals,
                np.array(dist).mean(axis=1),
                yerr=np.array(dist).std(axis=1),
                label=order,
                c=colour,
                lw=1.5,
                ls="dotted" if order == "Independent" else "-",
            )
            plt.title("Model Sensitivity (Ass. 2)")

            # Determine x-ticks and labels based on the number of layers.
            layers_str = [l.replace("model.network.", "layer.") for l in layers]
            if len(layers) > 20:
                xticks_indices = np.arange(0, len(layers), 3)  # Only every 3rd tick
            else:
                xticks_indices = np.arange(len(layers))  # All ticks

            # Labels for the selected ticks
            xticks_labels = [
                layers_str[i].replace("layer", "l").replace("downsample", "ds")
                for i in xticks_indices
            ]
            ax.set_xticks(xticks_indices)
            ax.set_xticklabels(
                xticks_labels[::-1] if order == "Bottom" else xticks_labels,
                rotation="vertical",
                fontsize=8 if "ResNet" in s else 10,
            )
        plt.legend(title="Order")
        plt.grid(True)
        plt.ylabel("Model Distortion")
        plt.xlabel("Layer-by-Layer Randomisation")
        plt.savefig(
            f'plots/model_sensitivity_analysis_{s.lower().replace("(", "").replace(", ", "_").replace(")", "")}.svg'
        )
        plt.show()


def plot_Z_influence(discovery_df, noise_types, Z):

    for noise_type in noise_types:
        settings = discovery_df.Setting.unique()
        fig, ax = plt.subplots(figsize=(4, 3))

        for s in settings:
            df_plot = discovery_df[discovery_df["Metric"].str.contains(noise_type)]
            discovery_df_setting = df_plot.loc[df_plot.Setting == s]

            M = 5
            nr_samples = discovery_df_setting.sample_size.iloc[0]
            perturbation_levels = np.arange(0, Z)
            colour = colours_discovery[s]

            xai_methods = len(discovery_df_setting)
            model_distortions = (
                np.array(
                    [
                        np.array(discovery_df_setting.model_distortions.iloc[i]).mean(
                            axis=0
                        )
                        for i in range(xai_methods)
                    ]
                )
                .transpose(1, 2, 0)
                .reshape(Z, nr_samples * xai_methods)
            )

            max_y_value = model_distortions.max()

            model_distortions_normalised = model_distortions / max_y_value

            ax.errorbar(
                x=perturbation_levels,
                y=model_distortions_normalised.mean(axis=1),
                yerr=model_distortions_normalised.std(axis=1),
                lw=1,
                marker="o",
                linestyle="dashed",
                label=s,
                color=colour,
            )

        plt.title(f"Influence of z={Z}")
        ax.set_xlabel(f"Pertubation levels")
        ax.set_ylabel("Model Distortion\n(normalised)")
        ax.set_xticks(np.arange(Z))
        ax.set_xticklabels(
            ["Acc. Orig"] + np.repeat("", Z - 2).tolist() + ["Acc. = 1/C"]
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncols=1)
        plt.grid(True)
        plt.savefig(f"plots/influence_Z={Z}_all_settings.svg")
        plt.show()


def plot_kde(df, methods, palette, colors_text, y, corr_method, suffix, mode):

    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.kdeplot(
        data=df[df["XAI Method"].isin(methods)],
        x=y,
        hue="XAI Method",
        hue_order=methods,
        fill=True,
        palette=palette,
        common_norm=False,
        alpha=0.5,
        bw_adjust=1.5,
        edgecolor="black",
        ax=ax,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors_text[key], edgecolor="black")
        for key in colors_text
    ]
    labels = methods
    ax.legend(
        handles, labels, title="Methods", loc="center left", bbox_to_anchor=(1, 0.5)
    )

    plt.title(f"Top-{suffix} Tokens")
    if mode == "naive":
        plt.xlabel(f"Fast-GEF Score")
    else:
        plt.xlabel(f"GEF Score")
    plt.grid(True)
    plt.savefig(
        f"plots/benchmarking_text_kde_{suffix}_{mode}.svg",
        facecolor=fig.get_facecolor(),
    )
    plt.show()


def prepare_text_results_naive(
    files_5_tokens=[
        "llmx_5_sst2_naive.pkl",
        "llmx_5_sms_naive.pkl",
        "llmx_text_control_top_5_naive.pkl",
    ],
    files_10_tokens=[
        "llmx_10_sst2_naive.pkl",
        "llmx_10_sms_naive.pkl",
        "llmx_text_control_top_10_naive.pkl",
    ],
    files_control=["llmx_text_control_top_5_10_naive.pkl"],
):

    # Process files.
    results_dfs_5s = [convert_dict_to_df(f, method_suffix="-5") for f in files_5_tokens]
    results_dfs_10s = [
        convert_dict_to_df(f, method_suffix="-10") for f in files_10_tokens
    ]
    results_dfs_ran = [convert_dict_to_df(f) for f in files_control]

    # Get master df.
    results_text_df = pd.concat(results_dfs_5s + results_dfs_10s + results_dfs_ran)
    results_text_df.reset_index(inplace=True)

    # Explode the DataFrame to flatten the lists in 'gef_scores'.
    results_text_df["gef_scores_spearmanr"] = results_text_df.apply(
        lambda row: recalculate_similarity_scores_row(row, similarity_func=spearmanr),
        axis=1,
    )
    results_text_df_ex = results_text_df.explode("gef_scores_spearmanr")
    results_text_df_ex["gef_scores_spearmanr_all"] = results_text_df_ex[
        "gef_scores_spearmanr"
    ].astype(float)
    results_text_df_ex.index = np.arange(len(results_text_df_ex))
    results_text_df_ex.sort_values(by="XAI Method", inplace=True)

    return results_text_df_ex


def plot_text_results_naive(results_text_df_ex):

    # Define colors.
    colors_text = {
        "LLM-x-5": "#f3b76d",  # f8e860
        "LLM-x-10": "#e69d5e",  # e4c22a
        "RAN-5": "gray",
        "RAN-10": "gray",
        "L-INTG-5": "#97d4cd",  # 7c95d9
        "L-INTG-10": "#67ae9a",  # 5c79c6
        "SHAP-P-5": "#6e9b90",  # 3d5eb3
        "SHAP-P-10": "#4b6962",  # 1f419f
    }

    # Create palette and order list.
    palette = list(colors_text.values())
    palettes = [item for item in palette for _ in range(2)]
    palettes = palette + palette
    dataset = "both"
    order = [
        "LLM-x-5",
        "LLM-x-10",
        "RAN-5",
        "RAN-10",
        "L-INTG-5",
        "L-INTG-10",
        "SHAP-P-5",
        "SHAP-P-10",
    ]

    # Plot for each Dataset in results_text_df_ex.
    dfs = {
        "sst2": results_text_df_ex.loc[results_text_df_ex.Dataset == "sst2"],
        "sms_spam": results_text_df_ex.loc[results_text_df_ex.Dataset == "sms_spam"],
    }

    for y in ["gef_scores_spearmanr_all"]:
        corr_method = y.split("_")[2].capitalize()
        fig, ax = plt.subplots(figsize=(5, 3))
        box = sns.boxplot(
            data=results_text_df_ex,
            x="XAI Method",
            hue="Dataset",
            y=y,
            order=order,
            ax=ax,
            showfliers=False,
        )

        # Apply hatching to dataset 2.
        num_boxes = len(results_text_df_ex["XAI Method"].unique())
        for i, patch in enumerate(box.patches):
            if i == len(colors_text) * len(dfs):
                break
            patch.set_facecolor(palettes[i])

        # Remove xlabel.
        ax.set_xlabel("")
        ax.set_ylabel(f"Fast-GEF Score")
        plt.grid(True)

        plt.xticks(fontsize=11, rotation=20)
        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                hatch="/" if key == "sms_spam" else "",
                edgecolor="black",
            )
            for key in dfs
        ]  # methods
        labels = dfs.keys()
        ax.legend(
            handles,
            labels,
            title="Datasets",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        plt.title(f"LLM-x vs Random vs Local")
        plt.savefig(f"plots/benchmarking_text_boxplots_{dataset}_naive.svg")
        plt.show()

    plot_kde(
        results_text_df_ex, order, palette, colors_text, y, corr_method, "K", "naive"
    )

    plot_kde(
        results_text_df_ex,
        ["LLM-x-5", "RAN-5", "L-INTG-5", "SHAP-P-5"],
        [
            colors_text["LLM-x-5"],
            colors_text["RAN-5"],
            colors_text["L-INTG-5"],
            colors_text["SHAP-P-5"],
        ],
        colors_text,
        y,
        corr_method,
        "5",
        "naive",
    )

    plot_kde(
        results_text_df_ex,
        ["LLM-x-10", "RAN-10", "L-INTG-10", "SHAP-P-10"],
        [
            colors_text["LLM-x-10"],
            colors_text["RAN-10"],
            colors_text["L-INTG-10"],
            colors_text["SHAP-P-10"],
        ],
        colors_text,
        y,
        corr_method,
        "10",
        "naive",
    )


def prepare_text_results_exact(
    files_5_tokens=[
        "llmx_exact_sms_5.pkl",
        "llmx_exact_sms_control_5.pkl",
        "llmx_exact_sst2_control_5.pkl",
        "llmx_exact_sst2_5.pkl",
        "llmx_exact_sst2_pshap_5.pkl",
        "llmx_exact_sms_pshap_5.pkl",
    ],
    files_10_tokens=[
        "llmx_exact_sms_control_10.pkl",
        "llmx_exact_sst2_10.pkl",
        "llmx_exact_sms_10.pkl",
        "llmx_exact_sst2_control_10.pkl",
        "llmx_exact_sst2_pshap_10.pkl",
        "llmx_exact_sms_pshap_10.pkl",
    ],
):

    # Process files.
    if files_5_tokens:
        results_dfs_5s = [
            convert_dict_to_df(f, method_suffix="-5") for f in files_5_tokens
        ]
    else:
        results_dfs_5s = []
    if files_10_tokens:
        results_dfs_10s = [
            convert_dict_to_df(f, method_suffix="-10") for f in files_10_tokens
        ]
    else:
        results_dfs_10s = []

    # Get master df.
    results_text_df = pd.concat(results_dfs_5s + results_dfs_10s)
    results_text_df.reset_index(inplace=True)

    # Explode the DataFrame to flatten the lists in 'gef_scores'.
    results_text_df["gef_scores_spearmanr"] = results_text_df.apply(
        lambda row: recalculate_similarity_scores_row(row, similarity_func=spearmanr),
        axis=1,
    )
    results_text_df_ex = results_text_df.explode("gef_scores_spearmanr")
    results_text_df_ex["gef_scores_spearmanr_all"] = results_text_df_ex[
        "gef_scores_spearmanr"
    ].astype(float)
    results_text_df_ex.index = np.arange(len(results_text_df_ex))
    results_text_df_ex.sort_values(by="XAI Method", inplace=True)

    return results_text_df_ex


def plot_text_results_exact(results_text_df_ex):

    # Define colors.
    colors_text = {
        "LLM-x-5": "#f3b76d",  # f8e860
        "LLM-x-10": "#e69d5e",  # e4c22a
        "RAN-5": "gray",
        "RAN-10": "gray",
        "L-INTG-5": "#97d4cd",  # 7c95d9
        "L-INTG-10": "#67ae9a",  # 5c79c6
        "SHAP-P-5": "#6e9b90",  # 3d5eb3
        #'SHAP-P-10': "#4b6962", #1f419f
    }

    # Create palette and order list.
    palette = list(colors_text.values())
    palettes = [item for item in palette for _ in range(2)]
    palettes = palette + palette
    dataset = "both"
    order = [
        "LLM-x-5",
        "LLM-x-10",
        "RAN-5",
        "RAN-10",
        "L-INTG-5",
        "L-INTG-10",
        "SHAP-P-5",
    ]  # , , 'SHAP-P-10']

    # Plot for each Dataset in results_text_df_ex.
    dfs = {
        "sst2": results_text_df_ex.loc[results_text_df_ex.Dataset == "sst2"],
        "sms_spam": results_text_df_ex.loc[results_text_df_ex.Dataset == "sms_spam"],
    }

    for y in ["gef_scores_spearmanr_all"]:
        corr_method = y.split("_")[2].capitalize()
        fig, ax = plt.subplots(figsize=(5, 3))
        box = sns.boxplot(
            data=results_text_df_ex,
            x="XAI Method",
            hue="Dataset",
            y=y,
            order=order,
            ax=ax,
            showfliers=False,
        )

        # Apply hatching to dataset 2.
        num_boxes = len(results_text_df_ex["XAI Method"].unique())
        for i, patch in enumerate(box.patches):
            if i == len(colors_text) * len(dfs):
                break
            patch.set_facecolor(palettes[i])

        # Remove xlabel.
        ax.set_xlabel("")
        ax.set_ylabel(f"GEF Score")
        plt.grid(True)

        plt.xticks(fontsize=11, rotation=20)
        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                hatch="/" if key == "sms_spam" else "",
                edgecolor="black",
            )
            for key in dfs
        ]  # methods
        labels = dfs.keys()
        ax.legend(
            handles,
            labels,
            title="Datasets",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        plt.title(f"LLM-x vs Random vs Local")
        plt.savefig(f"plots/benchmarking_text_boxplots_{dataset}_exact.svg")
        plt.show()

    plot_kde(
        results_text_df_ex, order, palette, colors_text, y, corr_method, "K", "exact"
    )

    plot_kde(
        results_text_df_ex,
        ["LLM-x-5", "RAN-5", "L-INTG-5", "SHAP-P-5"],  # ,
        [colors_text["LLM-x-5"], colors_text["RAN-5"], colors_text["L-INTG-5"]],  # ,
        colors_text,
        y,
        corr_method,
        "5",
        "exact",
    )

    plot_kde(
        results_text_df_ex,
        ["LLM-x-10", "RAN-10", "L-INTG-10"],  # , 'SHAP-P-10'
        [
            colors_text["LLM-x-10"],
            colors_text["RAN-10"],
            colors_text["L-INTG-10"],
        ],  # , colors_text['SHAP-P-10']
        colors_text,
        y,
        corr_method,
        "10",
        "exact",
    )


def plot_percentage_improvement_to_random(results_text_df_ex, colors_text):
    ran_avg = results_text_df_ex[
        results_text_df_ex["XAI Method"].isin(["RAN-5", "RAN-10"])
    ]["gef_scores_spearmanr_all"].mean()
    results_text_df_ex["Percentage Improvement"] = (
        (results_text_df_ex["gef_scores_spearmanr_all"] / ran_avg) - 1
    ) * 100
    results_text_df_ex["Base XAI Method"] = results_text_df_ex[
        "XAI Method"
    ].str.replace(r"-(5|10)", "", regex=True)
    grouped_mean = results_text_df_ex.groupby(["Base XAI Method"])[
        "Percentage Improvement"
    ].mean()
    grouped_std = results_text_df_ex.groupby(["Base XAI Method"])[
        "Percentage Improvement"
    ].std() / np.sqrt(250)

    base_colors_text = {
        "LLM-x": "#f3b76d",
        "L-INTG": "#97d4cd",
        "SHAP-P": "#6e9b90",
        "RAN": "gray",
    }

    fig, ax = plt.subplots(figsize=(3, 3))

    for method in ["LLM-x", "L-INTG", "SHAP-P"]:
        mean_value = grouped_mean.loc[method]
        std_value = grouped_std.loc[method]
        ax.barh(
            method,
            mean_value,
            xerr=std_value,
            color=base_colors_text.get(method, "gray"),
            edgecolor="black",
        )

    ax.set_xlabel("Relative Improvement (%)")
    ax.set_title("Relative GEF to Random")
    ax.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(f"plots/benchmarking_text_percentage_improvement.svg")
    plt.show()


def prepare_vision_tabular_results(file_names, file_names_100s, file_names_250s):

    # Process.
    results_dfs = [convert_dict_to_df(f, method_suffix=None) for f in file_names]
    results_dfs_100s = [
        convert_dict_to_df(f, method_suffix="-100") for f in file_names_100s
    ]
    results_dfs_250s = [
        convert_dict_to_df(f, method_suffix="-250") for f in file_names_250s
    ]

    # Get master df.
    results_df = pd.concat(results_dfs + results_dfs_100s + results_dfs_250s)
    results_df.reset_index(inplace=True)

    # Recompute rank and sort.
    del results_df["rank"]
    results_df["rank"] = results_df.groupby(["Setting", "Metric"])["mean_score"].rank(
        ascending=False
    )
    results_df.sort_values(by=["Setting", "Scope"], inplace=True)
    results_df.head(10)
    return results_df


def plot_bar_vision_tabular(results_df):

    # Assuming results_df is defined and contains the necessary data-
    BOTH_VARIANTS = True
    settings_separated = {
        "tabular": [
            "(Avila, 2-layer MLP)",
            "(Adult, 3-layer MLP)",
            "(Adult, LR)",
            "(COMPAS, 3-layer MLP)",
            "(COMPAS, LR)",
        ],
        "vision": [
            "(ImageNet, ResNet18)",
            "(Derma, MedCNN)",
            "(Path, MedCNN)",
            "(MNIST, LeNet)",
            "(fMNIST, LeNet)",
        ],
    }
    STD_TYPE = "std_error_score"
    for k, settings in settings_separated.items():

        # Determine the total number of XAI Methods across all settings.
        fig, ax = plt.subplots(figsize=(11, 2))
        bar_width = 0.2
        opacity = 0.75
        all_xai_methods = results_df["xai method"].unique()
        gap = 0.25
        base_x = 0
        full_x = []
        for i, setting in enumerate(settings):
            setting_data_filter = results_df[results_df["Setting"] == setting]

            if BOTH_VARIANTS:
                if not setting_data_filter.index.empty:
                    if setting_data_filter.Task.iloc[0] == "tabular":
                        setting_data_filter = setting_data_filter.loc[
                            (results_df["Metric"] == METHOD_NAME_EXACT)
                        ]
                    else:
                        setting_data_filter = setting_data_filter.loc[
                            (results_df["Metric"] == METHOD_NAME_NAIVE)
                        ]
            else:
                setting_data_filter = setting_data_filter.loc[
                    (results_df["Metric"] == METHOD_NAME_NAIVE)
                ]

            setting_data_filter = setting_data_filter.sort_values(
                by=["Scope", "xai method"]
            )
            xai_methods = setting_data_filter["xai method"].unique()
            x = np.arange(len(xai_methods)) * bar_width + base_x + (i * bar_width)
            methods = [c for c in colors if c in xai_methods]
            for ix, xai in enumerate(methods):
                mean_score = setting_data_filter.loc[
                    setting_data_filter["xai method"] == xai, "mean_score"
                ]
                std_score = setting_data_filter.loc[
                    setting_data_filter["xai method"] == xai, STD_TYPE
                ]
                plt.bar(
                    x[ix],
                    mean_score,
                    bar_width,
                    edgecolor="black",
                    yerr=std_score,
                    alpha=opacity,
                    color=colors[xai],
                    label=f"{xai}" if i == 0 else "",
                )
            base_x += gap + len(xai_methods) * bar_width
            full_x += [x]

        if STD_TYPE == "std_score":
            plt.ylim(-0.25, 1.0)
        else:
            plt.ylim(-0.05, 1.0)

        # Customising the plot.
        plt.ylabel("GEF Score")
        if k == "vision":
            plt.ylabel("Fast-GEF Score")
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=colors[key], alpha=opacity, edgecolor="black"
            )
            for key in colors
            if key != "model"
        ]
        labels = [key for key in colors if key not in TEXT_METHODS and key != "model"]
        plt.title("Cross-Domain Benchmarking of Global and Local Methods")
        plt.legend(
            handles,
            labels,
            title="Global and Local Methods",
            ncol=7,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.15),
        )
        x_tick_positions = [np.mean(x) for x in full_x]
        plt.xticks(
            x_tick_positions,
            labels=[s.replace("(", "").replace(", ", "\n(") for s in settings],
        ),
        plt.grid(True)
        plt.savefig(f"plots/gef_bar_{k}_{STD_TYPE}.svg", format="svg")
        plt.show()


def prepare_table(results_df_table, std_type: str = "std_error_score"):

    results_df_table = results_df_table.groupby(["dataset", "model", "xai method"])[
        ["mean_score", std_type]
    ].mean()
    table = (
        results_df_table.apply(
            lambda row: f'{row["mean_score"]:.2f} $\pm$ {row[std_type]:.2f}', axis=1
        )
        .unstack()
        .T
    )
    table.fillna("-", inplace=True)
    table = table.reset_index()
    table["Scope"] = table["xai method"].apply(
        lambda x: "Global" if x in GLOBAL_METHODS else "Local"
    )
    table = table.sort_values(by=["Scope", "xai method"], ascending=False)
    table = table.set_index("Scope").reset_index().set_index(["Scope", "xai method"])
    table.columns = table.columns.set_levels(
        [
            table.columns.levels[0]
            .str.capitalize()
            .map(lambda x: rf"\texttt{{{x}}}"),  # dataset
            table.columns.levels[1]
            .str.upper()
            .map(lambda x: rf"\texttt{{{x}}}"),  # model
        ]
    )
    table.index = table.index.set_levels(
        [
            table.index.levels[0]
            .str.capitalize()
            .map(lambda x: rf"\texttt{{{x}}}"),  # scope
            table.index.levels[1]
            .str.upper()
            .map(lambda x: rf"\texttt{{{x}}}"),  # xai method
        ]
    )
    return table


def postprocess_meta_evaluation(
    benchmark_m1,
    benchmark_m2,
    benchmark_f1,
    benchmark_f2,
    benchmark_im1,
    estimators_faith,
    estimators_not_exact,
):

    def process_df(df, dataset_name: str, xai_group: str):
        df["Category"] = pd.Categorical(
            df["Category"], categories=category_order_meta_evaluation, ordered=True
        )
        df["Estimator"] = df["Estimator"].map(abbreviations_meta_evaluation)
        df = df.sort_values(["Category", "Estimator"])
        df["dataset"] = dataset_name
        df["XAI-Group"] = xai_group
        return df

    # Load data.
    df_m1 = process_df(
        make_benchmarking_df(benchmark=benchmark_m1, estimators=estimators_faith),
        dataset_name="MNIST",
        xai_group="GS_SA",
    )
    df_m2 = process_df(
        make_benchmarking_df(benchmark=benchmark_m2, estimators=estimators_faith),
        dataset_name="MNIST",
        xai_group="G_GC",
    )
    df_f1 = process_df(
        make_benchmarking_df(benchmark=benchmark_f1, estimators=estimators_faith),
        dataset_name="fMNIST",
        xai_group="GS_SA",
    )
    df_f2 = process_df(
        make_benchmarking_df(benchmark=benchmark_f2, estimators=estimators_faith),
        dataset_name="fMNIST",
        xai_group="G_GC",
    )
    df_im1 = process_df(
        make_benchmarking_df(benchmark=benchmark_im1, estimators=estimators_not_exact),
        dataset_name="ImageNet",
        xai_group="G_GC_GS_SA",
    )
    df = pd.concat([df_m1, df_m2, df_f1, df_f2, df_im1])
    return df


def plot_meta_evaluation_bar(df, col_group: str = "Category", figsize=(11, 3)):
    hatch_patterns = {"MNIST": "/", "fMNIST": "", "ImageNet": "*"}
    hatch_patterns = {"Category": "/", "Estimator": ""}
    plt.figure(figsize=figsize)

    if col_group == "Estimator":
        df["Estimator"] = pd.Categorical(
            df["Estimator"],
            categories=[
                METHOD_NAME_EXACT,
                METHOD_NAME_NAIVE,
                "PF",
                "FC",
                "RP",
                "MPRT",
                "sMPRT",
                "eMPRT",
                "RIS",
                "ROS",
                "RRS",
            ],
            ordered=True,
        )

    for group_name, df_group in df.groupby([col_group]):

        label_name = group_name
        if "Fast" in group_name[0]:
            label_name = f"Fast\nGEF"

        bar = plt.bar(
            group_name,
            df_group["MC"].mean(),
            color=colours_meta_evalaution[df_group["Category"].unique()[0]],
            label=label_name,
            yerr=df_group["MC std"].mean(),
            capsize=5,
            alpha=0.6,
            edgecolor="black",
        )

        plt.text(
            bar[0].get_x() + bar[0].get_width() / 2,
            bar[0].get_height() + df_group["MC std"].mean() + 0.025,
            f'{df_group["MC"].mean():.3f}',
            ha="center",
            fontsize=11,
            va="bottom",
        )

    plt.ylabel(f"Mean MC (MPT, IPT)")
    plt.ylim([0.0, 1.0])
    if col_group == "Category":
        plt.legend(title=col_group, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        pass

    plt.grid(True)
    plt.title(f"Meta-Evaluation")
    plt.savefig(f"plots/meta_eval_{col_group}.svg")
    plt.show()


def plot_meta_evaluation_scatter(df):
    iters = 3
    K = 5
    sample_size = 250 * iters * K
    FACTOR = 2000
    sample_size = 250
    col_group = "Category"
    tests = ["model", "Input"]
    marker_styles = {"ImageNet": "o", "fMNIST": "s", "MNIST": "^"}  # , 'D', 'v']
    datasets = marker_styles.keys()
    fig, ax = plt.subplots(figsize=(3, 3))
    for group_name, df_group in df.groupby([col_group]):
        for dataset in datasets:

            ipt = df_group.loc[
                ((df_group.Test == "Input") & (df_group.Dataset == dataset)), "MC"
            ].mean()
            mpt = df_group.loc[
                ((df_group.Test == "model") & (df_group.Dataset == dataset)), "MC"
            ].mean()
            c = colours_meta_evalaution[df_group["Category"].unique()[0]]
            s = (
                (
                    df_group.loc[df_group.Test == "model", "MC std"].mean()
                    / np.sqrt(sample_size)
                )
                * FACTOR
                * 20
            )
            ax.scatter(
                ipt,
                mpt,
                color=c,
                label=dataset,
                s=s,
                alpha=0.6,
                marker=marker_styles[dataset],
                edgecolor="black",
            )
            ax.set_ylim(0.5, 1.0)
            ax.set_xlim(0.5, 1.0)

    plt.xlabel("MC (IPT)")
    plt.ylabel("Meta-Consistency (MPT)")
    plt.title(f"By Dataset")
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="white",
            markeredgecolor="black",
            marker=marker_styles[ds],
            linestyle="None",
            markersize=10,
            label=ds,
        )
        for ds in datasets
    ]
    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncols=1,
        title="dataset",
    )
    plt.grid(True)
    plt.savefig(f"plots/meta_eval_scatter.svg")
    plt.show()


def plot_global_local_distribution(results_df):

    # Explode the DataFrame to flatten the lists in 'gef_scores'
    exploded_df = results_df.explode("gef_scores")
    exploded_df["gef_scores"] = exploded_df["gef_scores"].astype(float)
    colors_loc_glob = {
        "Global": sns.color_palette("BuGn", 1).as_hex()[0],
        "Local": sns.color_palette("Purples", 1).as_hex()[0],
        "RAN": "gray",
    }
    exploded_df = exploded_df.loc[
        (exploded_df.Task == "vision") & (exploded_df.Metric == METHOD_NAME_NAIVE)
    ]
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.kdeplot(
        data=exploded_df[exploded_df["Scope"] == "Global"],
        x="gef_scores",
        fill=True,
        color=colors_loc_glob["Global"],
        alpha=0.75,
        bw_adjust=0.5,
        common_norm=False,
        edgecolor="black",
        label="Global",
    )
    sns.kdeplot(
        data=exploded_df[exploded_df["Scope"] == "Local"],
        x="gef_scores",
        fill=True,
        color=colors_loc_glob["Local"],
        alpha=0.75,
        bw_adjust=0.5,
        common_norm=False,
        edgecolor="black",
        label="Local",
    )

    ax.set_xlabel("GEF Score")
    ax.set_ylabel("Density")
    plt.legend(title="Scope", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.title("Global vs Local Methods")
    plt.savefig(f"plots/distribution_local_vs_global.svg")
    plt.show()


def plot_both_distortions(results_df, setting):

    results_df_setting = results_df.loc[results_df["Setting"] == setting]

    # Compute model distortions.
    dist_fs = []
    for i, method in enumerate(results_df_setting["XAI Method"].unique()):
        dist_f = np.array(results_df_setting.model_distortions.iloc[i])

        dist_fs.append(dist_f)
    dist_f = np.array(dist_fs).mean(axis=0)

    fig = plt.figure(figsize=(4, 3))

    for i, method in enumerate(results_df_setting["XAI Method"].unique()):

        # Compute explanation distortions.

        if "-100" in method:
            continue
        if "-50" in method:
            continue
        dist_e = np.array(results_df_setting.explanation_distortions.iloc[i])

        yerr = np.arange(1, 6)
        plt.plot(dist_f.mean(axis=(0, 2)), dist_e.mean(axis=(0, 2)), c=colors[method])
        plt.scatter(
            dist_f.mean(axis=(0, 2)),
            dist_e.mean(axis=(0, 2)),
            label=method,
            s=yerr**3,
            edgecolor="black",
            c=colors[method],
        )

    plt.legend(ncols=5, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(setting.replace("(", "").replace(", ", " ("))
    plt.xlabel("Model Distortion")
    plt.ylabel("Explanation Distortion")
    plt.grid(True)
    plt.savefig(
        f'plots/vision_expl_distortions_{setting.lower().replace("(", "").replace(", ", "_").replace(")", "")}.svg'
    )
    plt.show()


def compute_improvement_by_dataset(df_text):
    ran_avg = df_text[df_text["XAI Method"].isin(["RAN-5", "RAN-10"])][
        "gef_scores_spearmanr_all"
    ].mean()
    df_text["Percentage Improvement"] = (
        (df_text["gef_scores_spearmanr_all"] / ran_avg) - 1
    ) * 100
    df_text["Base XAI Method"] = df_text["XAI Method"].str.replace(
        r"-(5|10)", "", regex=True
    )

    datasets = df_text.Dataset.unique()
    methods = ["LLM-x", "L-INTG", "SHAP-P"]

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        for method in methods:
            data = df_text[
                (df_text["Dataset"] == dataset) & (df_text["Base XAI Method"] == method)
            ]
            mean_value = data["Percentage Improvement"].mean()
            std_value = data["Percentage Improvement"].std() / np.sqrt(len(data))
            print(f"{method}: {mean_value:.2f} ± {std_value:.2f}")
        print()


def plot_alignment_update(
    df_robustness,
    noise_types,
    noise_levels,
    title,
    colours_z,
    PLOT_CONTOUR: bool = True,
):
    """Plots alignment with consistent colours for Z levels and shared legend title."""

    colours_discovery = {
        "(COMPAS, 3-layer MLP)": "#81D46D",
        "(Avila, 2-layer MLP)": "#D8AD3A",
        "(ImageNet, ResNet18)": "#70A1CE",
        "(Path, MedCNN)": "lightcoral",
        "(MNIST, LeNet)": "sandybrown",
    }

    PLOT_CLIPPED = True

    Z = len(df_robustness.perturbation_path.iloc[0])
    M = len(df_gef_fast_gef.model_distortions[0])

    for noise_type in noise_types:
        df_plot = df_robustness[df_robustness["Metric"] == noise_type]

        nr_samples = df_plot.sample_size.iloc[0]

        settings_disc = df_plot.Setting.unique()
        perturbation_levels = np.arange(0, Z)

        for s in settings_disc:
            colour = colours_discovery[s]
            df_robustness_setting = df_plot.loc[df_plot.Setting == s]

            xai_methods = df_robustness_setting["XAI Method"].unique()

            for ix, xai in enumerate(xai_methods):
                df_robustness_setting_xai = df_robustness_setting.loc[
                    df_robustness_setting["XAI Method"] == xai
                ]
                model_distortions = np.array(
                    df_robustness_setting.model_distortions.iloc[ix]
                ).mean(axis=0)
                explanation_distortions = np.array(
                    df_robustness_setting.explanation_distortions.iloc[ix]
                ).mean(axis=0)

                mean_score = df_robustness_setting["mean_score"].values
                print(mean_score)

                if PLOT_CLIPPED:
                    model_distortions = np.clip(
                        model_distortions,
                        np.percentile(model_distortions, 1),
                        np.percentile(model_distortions, 99),
                    )
                    explanation_distortions = np.clip(
                        explanation_distortions,
                        np.percentile(explanation_distortions, 1),
                        np.percentile(explanation_distortions, 99),
                    )

                fig, ax = plt.subplots(figsize=(4, 3))

                for i in noise_levels:
                    if i in colours_z:
                        colour = colours_z[i]

                        if PLOT_CONTOUR:
                            xi = model_distortions[i]
                            yi = explanation_distortions[i]
                            X, Y = np.meshgrid(
                                np.linspace(xi.min(), xi.max(), 100),
                                np.linspace(yi.min(), yi.max(), 100),
                            )
                            xy = np.vstack(
                                [model_distortions[i], explanation_distortions[i]]
                            )
                            kde = gaussian_kde(xy, bw_method=0.5)

                            positions = np.vstack([X.ravel(), Y.ravel()])
                            Z_values = np.reshape(kde(positions).T, X.shape)
                            cmap = ListedColormap([colour])
                            cp = ax.contour(
                                X, Y, Z_values, cmap=cmap, levels=4, alpha=0.6
                            )

                        plt.scatter(
                            x=model_distortions[i],
                            y=explanation_distortions[i],
                            c=colour,
                            alpha=0.5,
                            edgecolor="black",
                            # label=f"Z={i}",
                            label="",
                        )

                plt.legend(
                    loc="upper right",
                    title=(r"$\rho(s, \hat{s})$" + f"={np.mean(mean_score):.2f}"),
                )  # title+"—"+
                plt.xlim(model_distortions.min() * 1.05, model_distortions.max() * 1.05)
                plt.ylim(
                    explanation_distortions.min() * 1.05,
                    explanation_distortions.max() * 1.05,
                )
                plt.xlabel("Model Distortion")
                plt.ylabel(f"{xai} Distortion")
                plt.title(s.replace("(", "").replace(", ", " ("))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    f'plots/alignment_update_{xai}_{title.lower().replace(" ", "_")}_{s.lower().replace("(", "").replace(", ", "_").replace(")", "")}.svg'
                )
                plt.show()


from matplotlib.collections import LineCollection


def plot_sample_distortion_gradient(
    df_robustness, noise_type, samples_to_plot, cmap="viridis"
):
    """
    Plots distortions for individual samples across perturbation levels with gradient colouring on lines.

    Args:
        df_robustness: DataFrame containing robustness metrics.
        noise_type: Specific noise type to filter.
        samples_to_plot: Number of samples to visualise per plot.
        cmap: Name of the colour map for gradient colouring.
    """
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize

    df_plot = df_robustness[df_robustness["Metric"].str.contains(noise_type)]

    settings_disc = df_plot.Setting.unique()

    for s in settings_disc:
        df_robustness_setting = df_plot.loc[df_plot.Setting == s]

        xai_methods = df_robustness_setting["XAI Method"].unique()
        cmap_instance = get_cmap(cmap)

        for xai in xai_methods:
            df_robustness_setting_xai = df_robustness_setting.loc[
                df_robustness_setting["XAI Method"] == xai
            ]
            model_distortions_all = np.array(
                df_robustness_setting_xai["model_distortions"].tolist()
            ).mean(axis=(0, 1))
            explanation_distortions_all = np.array(
                df_robustness_setting_xai["explanation_distortions"].tolist()
            ).mean(axis=(0, 1))

            num_samples = min(samples_to_plot, model_distortions_all.shape[-1])
            perturbation_levels = model_distortions_all.shape[0]

            fig, ax = plt.subplots(figsize=(4, 3))

            num_samples_indices = np.random.randint(
                0, model_distortions_all.shape[1], num_samples
            )
            for sample_idx in num_samples_indices:  # range():
                model_distortions = model_distortions_all[:, sample_idx]
                explanation_distortions = explanation_distortions_all[:, sample_idx]

                # Scatter points for clarity.
                ax.plot(
                    np.arange(perturbation_levels),
                    model_distortions,
                    label="Sample" if sample_idx == num_samples_indices[0] else None,
                    # explanation_distortions,
                )
                ax.scatter(
                    np.arange(perturbation_levels),
                    model_distortions,
                    # explanation_distortions,
                    #########c=np.arange(1, perturbation_levels + 1),
                    #########cmap=cmap,
                    edgecolor="black",
                    s=30,
                    zorder=3,
                )

            # Add colour bar.
            #######sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            #######sm.set_array([])
            #######cbar = plt.colorbar(sm, ax=ax)
            #######cbar.set_label("Perturbation Level (Z)", rotation=270, labelpad=15)

            plt.legend()
            plt.xlabel("Perturbation Level")
            plt.ylabel("Model Distortion")
            plt.title(f"{s}")
            plt.grid(True)
            plt.savefig(
                f'plots/distortion_gradient_line_{xai}_{s.lower().replace("(", "").replace(", ", "_").replace(")", "")}.svg'
            )
            plt.show()


def plot_sample_distortion_gradient(
    df_robustness, noise_type, samples_to_plot, cmap="viridis"
):
    """
    Plots distortions for individual samples across perturbation levels with gradient colouring on lines.

    Args:
        df_robustness: DataFrame containing robustness metrics.
        noise_type: Specific noise type to filter.
        samples_to_plot: Number of samples to visualise per plot.
        cmap: Name of the colour map for gradient colouring.
    """
    df_plot = df_robustness[df_robustness["Metric"].str.contains(noise_type)]
    settings_disc = df_plot.Setting.unique()

    for s in settings_disc:
        df_robustness_setting = df_plot.loc[df_plot.Setting == s]
        xai_methods = df_robustness_setting["XAI Method"].unique()
        cmap_instance = get_cmap(cmap)

        for a_ix, xai in enumerate(xai_methods):
            if a_ix > 0:
                continue
            df_robustness_setting_xai = df_robustness_setting.loc[
                df_robustness_setting["XAI Method"] == xai
            ]
            model_distortions_all = np.array(
                df_robustness_setting_xai["model_distortions"].tolist()
            ).mean(axis=(0, 1))
            perturbation_levels = model_distortions_all.shape[0]

            fig, ax = plt.subplots(figsize=(4.5, 3))
            num_samples = min(samples_to_plot, model_distortions_all.shape[-1])
            sample_indices = np.random.randint(
                0, model_distortions_all.shape[1], num_samples
            )

            for sample_idx in sample_indices:
                model_distortions = model_distortions_all[:, sample_idx]

                # Prepare segments for gradient line
                points = np.array(
                    [np.arange(perturbation_levels), model_distortions]
                ).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                norm = Normalize(vmin=0, vmax=perturbation_levels - 1)
                lc = LineCollection(segments, cmap=cmap_instance, norm=norm)
                lc.set_array(np.arange(perturbation_levels - 1))
                lc.set_linewidth(2)
                ax.add_collection(lc)

                # Add scatter points for clarity
                ax.scatter(
                    np.arange(perturbation_levels),
                    model_distortions,
                    c=np.arange(perturbation_levels),
                    cmap=cmap,
                    edgecolor="black",
                    s=40,
                    zorder=3,
                )

            # Add colour bar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Perturbation Level", rotation=270, labelpad=15)

            plt.xlabel("Perturbation Level")
            plt.ylabel("Model Distortion")
            plt.title(f"{s}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                f'plots/distortion_gradient_line_{xai}_{s.lower().replace("(", "").replace(", ", "_").replace(")", "")}.svg'
            )
            plt.show()

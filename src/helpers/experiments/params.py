"""This module contains the settings for the comparison run."""

from src.helpers.quantus_ext.quantus_explain import explain_gef
from quantus import similarity_func
import quantus

from src.gef import GEF
from src.helpers.experiments.layer_metric import MetricLayerDistortion

EXPLAIN_FUNC = explain_gef  # quantus.explain
SAVE = True
VERBOSE = False

full_size = 250
batch_size = 250


METRICS = {
    "Fast-GEF": GEF(
        fast_mode=False,
        similarity_func=similarity_func.correlation_spearman,
    ),
    "GEF - Exact": GEF(
        fast_mode=True,
        similarity_func=similarity_func.correlation_spearman,
    ),
}

# FIXME. Dont work with verbose.
METRICS_LAYER = {
    "Metric Layer Distortion - Bottom": MetricLayerDistortion(
        layer_order="bottom_up",
    ),
    "Metric Layer Distortion - Top": MetricLayerDistortion(
        layer_order="top_down",
    ),
    "Metric Layer Distortion - Independent": MetricLayerDistortion(
        layer_order="independent",
    ),
}

METRICS_DISCOVERY = {
    "Fast-GEF - Input - Additive - Class": GEF(
        Z=20,
        fast_mode=True,
        input_mode=True,
        noise_type="additive",
        mean=0.0,
        evaluate_class_distortion=True,
    ),
    "GEF - Input - Multiplicative - Class": GEF(
        Z=20,
        fast_mode=False,
        input_mode=True,
        noise_type="multiplicative",
        mean=1.0,
        evaluate_class_distortion=True,
    ),
    "GEF - Model - Multiplicative - Class": GEF(
        Z=20,
        fast_mode=True,
        input_mode=False,
        noise_type="multiplicative",
        mean=1.0,
        evaluate_class_distortion=True,
    ),
}

METRICS_RANDOM = {
    "FC": quantus.FaithfulnessCorrelation(
        subset_size=28,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        nr_runs=5,
        abs=False,
        normalise=True,
        normalise_func=quantus.normalise_func.normalise_by_average_second_moment_estimate,
        return_aggregate=False,
        disable_warnings=True,
        display_progressbar=False,
    ),
    "MPRT": quantus.MPRT(
        skip_layers=True,
        return_last_correlation=True,
        similarity_func=similarity_func.correlation_spearman,
        abs=False,
        normalise=True,
        normalise_func=quantus.normalise_func.normalise_by_average_second_moment_estimate,
        return_aggregate=False,
        disable_warnings=True,
        display_progressbar=False,
    ),
    "RIS": quantus.RelativeInputStability(
        nr_samples=5,
        perturb_func=quantus.perturb_func.gaussian_noise,
        return_nan_when_prediction_changes=False,
        abs=False,
        normalise=True,
        normalise_func=quantus.normalise_func.normalise_by_average_second_moment_estimate,
        return_aggregate=False,
        disable_warnings=True,
        display_progressbar=False,
    ),
    "Fast-GEF": GEF(
        fast_mode=True,
        similarity_func=similarity_func.correlation_spearman,
        abs=False,
        normalise=True,
        normalise_func=quantus.normalise_func.normalise_by_average_second_moment_estimate,
        return_aggregate=False,
        disable_warnings=True,
        display_progressbar=False,
    ),
    "GEF": GEF(
        fast_mode=False,
        similarity_func=similarity_func.correlation_spearman,
        abs=False,
        normalise=True,
        normalise_func=quantus.normalise_func.normalise_by_average_second_moment_estimate,
        return_aggregate=False,
        disable_warnings=True,
        display_progressbar=False,
    ),
}


TASKS = [
    ("vision", "mnist", "lenet", full_size, batch_size),
    ("vision", "fashion_mnist", "lenet", full_size, batch_size),
    ("vision", "path", "cnn", full_size, batch_size),
    ("vision", "blood", "cnn", full_size, batch_size),
    ("vision", "derma", "cnn", full_size, batch_size),
    ("vision", "imagenet-1k", "torchvision.models.resnet18", full_size, batch_size),
    ("vision", "imagenet-1k", "torchvision.models.vit_b_16", full_size, batch_size),
    ("tabular", "adult", "ann", full_size, batch_size),
    ("tabular", "adult", "lr", full_size, batch_size),
    ("tabular", "compas", "ann", full_size, batch_size),
    ("tabular", "compas", "lr", full_size, batch_size),
    ("tabular", "avila", "mlp", full_size, batch_size),
    (
        "text",
        "sms_spam",
        "mrm8488/bert-tiny-finetuned-sms-spam-detection",
        full_size,
        50,
    ),
    (
        "text",
        "sms_spam",
        "mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection",
        full_size,
        50,
    ),
    (
        "text",
        "imdb",
        "AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3",
        full_size,
        50,
    ),
    ("text", "sst2", "VityaVitalich/bert-tiny-sst2", full_size, 125),
    (
        "text",
        "emotion",
        "j-hartmann/emotion-english-distilroberta-base",
        full_size,
        50,
    ),
]

XAI_METHODS_MAPPING = {
    "LLM": "LLM-x",
    "RA": "Random Guess",
    "RAK": "Random Guess K",
    "RAK-5": "Random Guess K=5",
    "RAK-10": "Random Guess K=10",
    "RAK-20": "Random Guess K=20",
    "DV": "Deep-Vis",
    "MACO": "MACO",
    "FO": "Fourier",
    "GRAD": "Gradient",
    "SAL": "Saliency",
    "IXG": "InputXGradient",
    "G-CAM": "LayerGradCam",
    "SMG": "SmoothGrad",
    "INT-G": "IntegratedGradients",
    "L-INTG": "LayerIntegratedGradients",
    "GBP": "Guided-Backprop",
    "SHAP-G": "GradientShap",
    "SHAP-P": "PartitionShap",
    "LRPeps": "LRP-Eps",
    "LRPz": "LRP-Z+",
    "OC": "Occlusion",
    "CVC": "Control Var. Constant",
    "CVS": "Control Var. Sobel Filter",
    "CVR": "Control Var. Random Uniform",
}

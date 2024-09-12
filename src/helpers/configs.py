"""This module contains the configuration for plotting."""

import json
from collections import OrderedDict
import numpy as np
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns


with zipfile.ZipFile("Roboto_Mono.zip", "r") as zip_ref:
    zip_ref.extractall(".")

font_dir = ["static/"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
plt.rcParams["font.family"] = "DejaVu Sans Mono"

available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
if "DejaVu Sans Mono" in available_fonts:
    print("DejaVu Sans Mono font successfully loaded.")
else:
    print("DejaVu Sans Mono font not found. Available fonts:", available_fonts)

# Set font size.
plt.rcParams["font.size"] = 11
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13

# Disable unicode minus.
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True

# Further modernize the plot appearance.
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.5


with open(f"meta_data_tasks.json", "rb") as f:
    META = json.load(f)

METHOD_NAME_NAIVE = "Fast-GEF"
METHOD_NAME_EXACT = "GEF"

# Post-process fixes.

# Post-process fixes.
REPLACE = {
    "Bridge - Naive": METHOD_NAME_NAIVE,
    "Bridge - Exact": METHOD_NAME_EXACT,
    "(imagenet-1k, torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT))": "(ImageNet, ResNet18)",
    "(imagenet-1k, torchvision.models.resnet18)": "(ImageNet, ResNet18)",
    "torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)": "ResNet18",
    "torchvision.models.resnet18": "ResNet18",
    "(sms_spam, mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection)": "(sms_spam, distilbert-FT)",
    "mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection": "distilbert-FT",
    "(sms_spam, mrm8488/bert-tiny-finetuned-sms-spam-detection)": "(sms_spam, BERT tiny-FT)",
    "mrm8488/bert-tiny-finetuned-sms-spam-detection": "BERT tiny-FT",
    "(sst2, VityaVitalich/bert-tiny-sst2)": "(SST2, BERT tiny-FT)",
    "VityaVitalich/bert-tiny-sst2": "BERT tiny-FT",
    "(imdb, AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3)": "(IMDb, Pythia-14M)",
    "AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3": "Pythia-14M",
    "imagenet-1k": "ImageNet",
    "imdb": "IMDb",
    "fasion_mnist": "fMNIST",
    "mnist": "MNIST",
    "lenet": "LeNet",
    "compas": "COMPAS",
    "derma": "Derma",
    "blood": "Blood",
    "path": "Path",
    "cnn": "MedCNN",
    "avila": "Avila",
    "mlp": "2-layer MLP",
    "ann": "3-layer MLP",
    "(fashion_mnist, lenet)": "(fMNIST, LeNet)",
    "(mnist, lenet)": "(MNIST, LeNet)",
    "(compas, lr)": "(COMPAS, LR)",
    "(adult, lr)": "(Adult, LR)",
    "(compas, ann)": "(COMPAS, 3-layer MLP)",
    "(adult, ann)": "(Adult, 3-layer MLP)",
    "(blood, cnn)": "(Blood, MedCNN)",
    "(avila, mlp)": "(Avila, 2-layer MLP)",
    "(derma, cnn)": "(Derma, MedCNN)",
    "(path, cnn)": "(Path, MedCNN)",
    "IntegratedGradients": "INTG",
    "GuidedBackprop": "GBP",
    "Guided-Backprop": "GBP",
    "GradientShap": "SHAP-G",
    "MACO": "MACO-50",
    "Fourier": "FO-50",
    "Act-Max": "DV-50",
    "MACO-100": "MACO-100",
    "Fourier-100": "FO-100",
    "Act-Max-100": "DV-100",
    "MACO-250": "MACO-250",
    "Fourier-250": "FO-250",
    "Act-Max-250": "DV-250",
    "LRP-Eps": "LRP-ε",
    "LRP-Z+": "LRP-z+",
    "Saliency": "SAL",
    "Gradient": "GRAD",
    "Auto-Interpret": "LLM-x",
    "IntegratedGradients": "INTG",
    "LayerIntegratedGradients": "L-INTG",
    "Random Guess": "RAN",
    "Random Guess K": "RAN-K",
    "Saliency": "SAL",
    "LayerGradCam": "G-CAM",
    "SmoothGrad": "SMG",
    "PartitionShap": "SHAP-P",
    "GradientShap": "SHAP-G",
    "PartitionShap-10": "SHAP-P-10",
    "PartitionShap-5": "SHAP-P-5",
    "InputXGradient": "IXG",
    "Auto-Interpret-5": "LLM-x-5",
    "Auto-Interpret-10": "LLM-x-10",
    "LayerIntegratedGradients-5": "L-INTG-5",
    "LayerIntegratedGradients-10": "L-INTG-10",
    "PartitionShap-1": "SHAP-P-5",
    "PartitionShap-1": "SHAP-P-10",
    "Random Guess K=5": "RAN-5",
    "Random Guess K=10": "RAN-10",
    # "Random Guess K=5-5": "RAN-5",
    # "Random Guess K=10-10": "RAN-10",
}

REPLACE_TEXT = {
    "Auto-Interpret-5": "LLM-x-5",
    "Auto-Interpret-10": "LLM-x-10",
    "LayerIntegratedGradients-5": "L-INTG-5",
    "LayerIntegratedGradients-10": "L-INTG-10",
    "PartitionShap-1": "SHAP-P-5",
    "PartitionShap-1": "SHAP-P-10",
    "Random Guess K=5": "RAN-5",
    "Random Guess K=10": "RAN-10",
}

GLOBAL_METHODS = [
    "DV-50",
    "DV-100",
    "DV-250",
    "FO-50",
    "FO-100",
    "FO-250",
    "MACO-50",
    "MACO-100",
    "MACO-250",
]
LOCAL_METHODS = [
    "GRAD",
    "SAL",
    "SHAP-G",
    "GBP",
    "IXG",
    "INTG",
    "SMG",
    "LRP-z+",
    "LRP-ε",
]
TEXT_METHODS = [
    "LLM-x-5",
    "LLM-x-10",
    "L-INTG-5",
    "L-INTG-10",
    "SHAP-P-5",
    "SHAP-P-10",
    "RAN-10",
    "RAN-5",
]
RANDOM_METHODS = ["RAN", "RAN-5", "RAN-10"]

# Define colors.
global_colors = dict(
    zip(GLOBAL_METHODS, (sns.color_palette("Greens", len(GLOBAL_METHODS)).as_hex()))
)
local_colors = dict(
    zip(LOCAL_METHODS, sns.color_palette("Purples", len(LOCAL_METHODS)).as_hex())
)
colors = {**global_colors, **local_colors}
for r in RANDOM_METHODS:
    colors[r] = "gray"
colors = OrderedDict((key, colors[key]) for key in colors)
colors = {
    **colors,
    **{
        "LLM-x-5": "#e1b9fc",
        "LLM-x-10": "#9c69cb",
        "L-INTG-5": "#476f95",
        "L-INTG-10": "#194a7a",
        "SHAP-P-5": "#6a8e8f",
        "SHAP-P-10": "#1c4e4f",
    },
}
colors_text = {
    "L-INTG-5": "#af92b5",
    "L-INTG-10": "#6f597a",
    "SHAP-P-5": "#bd3d3f",
    "SHAP-P-10": "#924144",
    "LLM-x-5": "#a3b7ca",
    "LLM-x-10": "#7593af",
    "RAN-5": "gray",
    "RAN-10": "gray",
}


colours_discovery = {
    "(COMPAS, 3-layer MLP)": "#81D46D",
    "(Avila, 2-layer MLP)": "#D8AD3A",
    "(ImageNet, ResNet18)": "#70A1CE",
    "(Path, MedCNN)": "lightcoral",
    "(MNIST, LeNet)": "sandybrown",
}


category_order_meta_evaluation = [
    "Unified",
    "Faithfulness",
    "Sensitivity",
    "Robustness",
]
abbreviations_meta_evaluation = {
    "Pixel-Flipping": "PF",
    "Faithfulness Correlation": "FC",
    "Faithfulness Estimate": "FE",
    "Region Perturbation": "RP",
    "MPRT": "MPRT",
    "sMPRT": "sMPRT",
    "eMPRT": "eMPRT",
    "Relative Output Stability": "ROS",
    "Relative Representation Stability": "RRS",
    "Relative Input Stability": "RIS",
    "Bridge - Exact": METHOD_NAME_EXACT,
    "Bridge - Naive": METHOD_NAME_NAIVE,
}
colours_meta_evalaution = {
    "Unified": "#154c79",
    "Faithfulness": "#829c9e",
    "Sensitivity": "#92ba92",
    "Robustness": "#666b5e",
}

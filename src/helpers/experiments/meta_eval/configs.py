"""This module contains functions for setting up experiments, adapted from https://github.com/annahedstroem/sanity-checks-revisited."""

from typing import Callable, Optional, Dict, Any, Callable
import numpy as np
import quantus
from dataclasses import dataclass

from typing import Dict, Optional, List
import numpy as np
import torch
import torchvision
from zennit import canonizers, composites, rules, attribution
from zennit import torchvision as zvision
from zennit import types as ztypes
from quantus import AVAILABLE_XAI_METHODS_CAPTUM
from typing import Dict
import numpy as np
import torch
import torchvision
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights

from metaquantus import LeNet
from metaquantus import ModelPerturbationTest, InputPerturbationTest

from src.gef import GEF
from src.helpers.experiments.setup_experiments import (
    RetinaCNN,
    DermaCNN,
    PathCNN,
    BloodCNN,
    ChestCNN,
    PneumoniaCNN,
)


def setup_test_suite_gef(std_adversary: float = 2.0):
    return {
        "Model Resilience Test": ModelPerturbationTest(
            **{
                "noise_type": "multiplicative",
                "mean": 1.0,
                "std": 0.001,
                "type": "Resilience",
            }
        ),
        "Model Adversary Test": ModelPerturbationTest(
            **{
                "noise_type": "multiplicative",
                "mean": 1.0,
                "std": std_adversary,
                "type": "Adversary",
            }
        ),
        "Input Resilience Test": InputPerturbationTest(
            **{
                "noise": 0.001,
                "type": "Resilience",
            }
        ),
        "Input Adversary Test": InputPerturbationTest(
            **{
                "noise": 5.0,
                "type": "Adversary",
            }
        ),
    }


# Experimental setup.
XAI_METHODS_ALL = {
    "G_GC": ["Gradient", "LayerGradCam"],
    "SA_IG_IXG": ["Saliency", "IntegratedGradients", "InputXGradient"],
    "GS_SA": ["GradientShap", "Saliency"],
    "LRPplus_IXG_GX": ["LRP-Z+", "InputXGradient", "GradientShap"],
    "G_GC_LRPeps_GB": ["Gradient", "LayerGradCam", "LRP-Eps", "Guided-Backprop"],
    "GP_GS_GC_LRP-Eps_SA": [
        "Guided-Backprop",
        "GradientShap",
        "LayerGradCam",
        "LRP-Eps",
        "Saliency",
    ],
    "full_set": [
        "Gradient",
        "Saliency",
        "LayerGradCam",
        "SmoothGrad",
        "IntegratedGradients",
        "LRP-Eps",
        "LRP-Z+",
        "Guided-Backprop",
        "GradientShap",
        "InputXGradient",
    ],
}
DATASETS = {
    "fMNIST": {
        "model_name": "LeNet",
        "indices": [[0, 250]],
    },
    "MNIST": {
        "model_name": "LeNet",
        "indices": [[0, 250]],
    },
    "Retina": {
        "model_name": "CNN",
        "indices": [[0, 250]],
    },
    "Derma": {
        "model_name": "CNN",
        "indices": [[0, 250]],
    },
    "Path": {
        "model_name": "CNN",
        "indices": [[0, 250]],
    },
    "Blood": {
        "model_name": "CNN",
        "indices": [[0, 250]],
    },
    "Chest": {
        "model_name": "CNN",
        "indices": [[0, 250]],
    },
    "Pneumonia": {
        "model_name": "CNN",
        "indices": [[0, 250]],
    },
    "ImageNet": {
        "model_name": "ResNet18",
        "indices": [
            [0, 50],
            [50, 100],
            [100, 150],
            [150, 200],
            [200, 250],
        ],  # get_indices(batch_size=100, dataset_length=100),
    },
    # "ImageNet_VGG16": {
    #     "model_name": "VGG16",
    #     "indices": get_indices(batch_size=50, dataset_length=100),
    # },
}


def get_zennit_canonizer(model):
    """
    Checks the type of model and selects the corresponding zennit canonizer
    """

    # ResNet
    if isinstance(model, torchvision.models.ResNet):
        return zvision.ResNetCanonizer

    # VGG
    if isinstance(model, torchvision.models.VGG):
        return zvision.VGGCanonizer

    # default fallback (only the above types have specific canonizers in zennit for now)
    return canonizers.SequentialMergeBatchNorm


class Epsilon(composites.LayerMapComposite):
    """An explicit composite using the epsilon rule for all layers

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    """

    def __init__(
        self,
        epsilon=1e-6,
        stabilizer=1e-6,
        layer_map=None,
        zero_params=None,
        canonizers=None,
    ):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {"zero_params": zero_params}
        layer_map = (
            layer_map
            + composites.layer_map_base(stabilizer)
            + [
                (ztypes.Convolution, rules.Epsilon(epsilon=epsilon, **rule_kwargs)),
                (torch.nn.Linear, rules.Epsilon(epsilon=epsilon, **rule_kwargs)),
            ]
        )
        super().__init__(layer_map=layer_map, canonizers=canonizers)


class ZPlus(composites.LayerMapComposite):
    """
    An explicit composite using the epsilon rule for all layers

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    """

    def __init__(
        self, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None
    ):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {"zero_params": zero_params}
        layer_map = (
            layer_map
            + composites.layer_map_base(stabilizer)
            + [
                (
                    ztypes.Convolution,
                    rules.ZPlus(stabilizer=stabilizer, **rule_kwargs),
                ),
                (torch.nn.Linear, rules.ZPlus(stabilizer=stabilizer, **rule_kwargs)),
            ]
        )
        super().__init__(layer_map=layer_map, canonizers=canonizers)


def setup_xai_methods_captum(
    xai_methods: List[str],
    x_batch: np.array,
    gc_layer: Optional[str] = None,
    img_size: int = 28,
    nr_channels: int = 1,
    nr_segments: int = 25,
) -> Dict:

    captum_methods = {
        "Gradient": {},
        "Saliency": {},
        "DeepLift": {},
        "GradientShap": {},
        "InputXGradient": {},
        "IntegratedGradients": {"n_steps": 10},
        "LayerGradCam": {
            "gc_layer": gc_layer,
            "interpolate": (img_size, img_size),
            "interpolate_mode": "bilinear",
            "xai_lib": "captum",
        },
        "Occlusion": {
            "window": (nr_channels, int(img_size / 4), int(img_size / 4)),
            "xai_lib": "captum",
        },
    }

    return {
        xai: captum_methods.get(xai, {"xai_lib": "captum"})
        for xai in xai_methods
        if xai in captum_methods
    }


def setup_xai_methods_zennit(
    xai_methods: List[str], model: torch.nn.Module, device: Optional[str] = None
) -> Dict[str, dict]:

    zennit_methods = {
        "SmoothGrad": {
            "xai_lib": "zennit",
            "attributor": attribution.SmoothGrad,
            "attributor_kwargs": {"n_iter": 10, "noise_level": 0.1},
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "IntegratedGradients": {
            "xai_lib": "zennit",
            "attributor": attribution.IntegratedGradients,
            "attributor_kwargs": {
                "n_iter": 20,
            },
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "LRP-Eps": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "composite": Epsilon,
            "canonizer": get_zennit_canonizer(model),
            "canonizer_kwargs": {},
            "composite_kwargs": {"stabilizer": 1e-6, "epsilon": 1e-6},
            "device": device,
        },
        "LRP-Z+": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "composite": ZPlus,
            "canonizer": get_zennit_canonizer(model),
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
            },
            "device": device,
        },
        "Guided-Backprop": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "canonizer": None,
            "composite": composites.GuidedBackprop,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
    }

    return {xai: zennit_methods[xai] for xai in xai_methods if xai in zennit_methods}


def setup_tasks(
    dataset_name: str,
    path_assets: str,
    device: torch.device,
    suffix: str = "_random",
) -> Dict[str, dict]:
    """
    Setup dataset-specific models and data for MPRT, SmoothMPRT and EfficientMPRT evaluation.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset ('MNIST', 'fMNIST', 'ImageNet').

    path_assets : str
        The path to the assets directory containing models and test sets.

    device : torch.device
        The device to load the models onto (e.g., 'cpu' or 'cuda').

    suffix : str, optional
        Suffix for dataset-specific assets.

    Returns
    -------
    dict
        A dictionary containing settings for eMPRT evaluation.
    """

    SETTINGS = {}
    SETTINGS[dataset_name] = {}

    if dataset_name in "MNIST":
        # Paths.
        path_model = path_assets + f"models/mnist_lenet"
        path_assets = path_assets + f"test_sets/mnist_test_set.npy"

        # Load model.
        model = LeNet()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"LeNet": model},
            "gc_layers": {"LeNet": "list(model.named_modules())[3][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 10,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "perturb_baseline": "uniform",
                "std_adversary": 2.0,
                "layer_indices": [-2],
                "layer_names": None,
            },
        }

    elif dataset_name == "fMNIST":
        # Paths.
        path_model = path_assets + f"models/fmnist_lenet"
        path_assets = path_assets + f"test_sets/fmnist_test_set.npy"

        # Load model.
        model = LeNet()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"LeNet": model},
            "gc_layers": {"LeNet": "list(model.named_modules())[3][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 10,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "perturb_baseline": "uniform",
                "std_adversary": 2.0,
                "layer_indices": [-2],
                "layer_names": None,
            },
        }

    elif dataset_name == "Retina":

        # Paths.
        path_model = path_assets + f"models/retina_cnn"
        path_assets = path_assets + f"test_sets/retina_test_set.npy"

        # Load model.
        model = RetinaCNN()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"CNN": model},
            "gc_layers": {"CNN": "list(model.named_modules())[-11][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 5,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 3,
                "perturb_baseline": "uniform",
                "std_adversary": 4.5,
                "layer_indices": None,
                "layer_names": ["fc.2"],
            },
        }
    elif dataset_name == "Derma":

        # Paths.
        path_model = path_assets + f"models/derma_cnn.pth"
        path_assets = path_assets + f"test_sets/derma_test_set.npy"

        # Load model.
        model = DermaCNN()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"CNN": model},
            "gc_layers": {"CNN": "list(model.named_modules())[-11][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 7,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 3,
                "perturb_baseline": "uniform",
                "std_adversary": 4.5,
                "layer_indices": None,
                "layer_names": ["fc.2"],
            },
        }

    elif dataset_name == "Path":

        # Paths.
        path_model = path_assets + f"models/path_cnn.pth"
        path_assets = path_assets + f"test_sets/path_test_set.npy"

        # Load model.
        model = PathCNN()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"CNN": model},
            "gc_layers": {"CNN": "list(model.named_modules())[-11][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 9,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 3,
                "perturb_baseline": "uniform",
                "std_adversary": 4.5,
                "layer_indices": None,
                "layer_names": ["fc.2"],
            },
        }
    elif dataset_name == "Blood":

        # Paths.
        path_model = path_assets + f"models/blood_cnn.pth"
        path_assets = path_assets + f"test_sets/blood_test_set.npy"

        # Load model.
        model = BloodCNN()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"CNN": model},
            "gc_layers": {"CNN": "list(model.named_modules())[-11][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 8,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 3,
                "perturb_baseline": "uniform",
                "std_adversary": 1.5,
                "layer_indices": None,
                "layer_names": ["fc.2"],
            },
        }

    elif dataset_name == "Chest":

        # Paths.
        path_model = path_assets + f"models/chest_cnn.pth"
        path_assets = path_assets + f"test_sets/chest_test_set.npy"

        # Load model.
        model = ChestCNN()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"CNN": model},
            "gc_layers": {"CNN": "list(model.named_modules())[-11][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 14,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "perturb_baseline": "uniform",
                "std_adversary": 5.0,
                "layer_indices": None,
                "layer_names": ["fc.2"],
            },
        }

    elif dataset_name == "Pneumonia":

        # Paths.
        path_model = path_assets + f"models/pneumonia_cnn.pth"
        path_assets = path_assets + f"test_sets/pneumonia_test_set.npy"

        # Load model.
        model = PneumoniaCNN()
        model.load_state_dict(torch.load(path_model, map_location=device))

        # Load data.
        assets = np.load(path_assets, allow_pickle=True).item()
        x_batch = assets["x_batch"]
        y_batch = assets["y_batch"]
        s_batch = assets["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {"CNN": model},
            "gc_layers": {"CNN": "list(model.named_modules())[-11][1]"},
            "estimator_kwargs": {
                "features": 28,
                "patch_size": 7,
                "num_classes": 2,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "perturb_baseline": "uniform",
                "std_adversary": 2.75,
                "layer_indices": None,
                "layer_names": ["fc.2"],
            },
        }

    elif dataset_name == "ImageNet":
        # Paths.
        path_imagenet_assets = path_assets + f"test_sets/imagenet_test_set{suffix}.npy"

        # Load data.
        assets_imagenet = np.load(path_imagenet_assets, allow_pickle=True).item()
        x_batch = assets_imagenet["x_batch"]
        y_batch = assets_imagenet["y_batch"]
        s_batch = assets_imagenet["s_batch"]

        # Add to settings.
        SETTINGS[dataset_name] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "s_batch": s_batch,
            "models": {
                "ResNet18": torchvision.models.resnet18(
                    weights=ResNet18_Weights.DEFAULT
                ).eval(),
                "VGG16": torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval(),
            },
            "gc_layers": {
                "ResNet18": "list(model.named_modules())[61][1]",
                "VGG16": "model.features[-2]",
            },
            "estimator_kwargs": {
                "num_classes": 1000,
                "features": 224 * 4,
                "patch_size": 28,
                "img_size": 224,
                "percentage": 0.1,
                "nr_channels": 3,
                "perturb_baseline": "uniform",
                "std_adversary": 0.5,
                "layer_indices": None,  # [-2],
                "layer_names": ["avgpool"],
            },
        }

    else:
        raise ValueError(
            "Provide a supported dataset {'MNIST', 'fMNIST', 'ImageNet', 'Derma', 'Retina'}."
        )

    return SETTINGS


@dataclass
class Estimator:
    name: str
    category: str
    score_direction_lower_is_better: bool
    init: quantus.Metric


def setup_metrics_gef(
    gef_kwargs: dict,
    num_classes: int,
    similarity_func: Callable = quantus.similarity_func.correlation_spearman,
    abs: bool = False,
    normalise: bool = True,
    normalise_func: Callable = quantus.normalise_func.normalise_by_average_second_moment_estimate,
    return_aggregate: bool = False,
    disable_warnings: bool = True,
    display_progressbar: bool = False,
) -> Dict:

    # Define the estimators.
    estimators = [
        Estimator(
            name=f"GEF - Exact {name}",
            category="Unified",
            score_direction_lower_is_better=False,
            init=GEF(
                exact_mode=True,
                similarity_func=similarity_func,
                num_classes=num_classes,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                aggregate_func=np.mean,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
                **values,
            ),
        )
        for name, values in gef_kwargs.items()
    ]

    estimator_dict = {}
    for estimator in estimators:
        if estimator.category not in estimator_dict:
            estimator_dict[estimator.category] = {}
        estimator_dict[estimator.category][estimator.name] = {
            "init": estimator.init,
            "score_direction": estimator.score_direction_lower_is_better,
        }

    return estimator_dict


def setup_metrics_meta(
    M: int,
    Z: int,
    T: int,
    K: int,
    features: int,
    patch_size: int,
    num_classes: int,
    perturb_baseline: str = "uniform",
    x_noise: float = 0.01,
    layer_indices: Optional[List[int]] = None,
    layer_names: Optional[List[str]] = None,
    similarity_func: Callable = quantus.similarity_func.correlation_spearman,
    nr_samples: int = 5,
    abs: bool = False,
    normalise: bool = True,
    normalise_func: Callable = quantus.normalise_func.normalise_by_average_second_moment_estimate,
    perturb_func: Callable = quantus.perturb_func.baseline_replacement_by_indices,
    return_aggregate: bool = False,
    disable_warnings: bool = True,
    display_progressbar: bool = False,
) -> Dict:

    # Define the estimators.
    estimators = [
        Estimator(
            name="Relative Representation Stability",
            category="Robustness",
            score_direction_lower_is_better=True,
            init=quantus.RelativeRepresentationStability(
                nr_samples=nr_samples,
                layer_indices=layer_indices,
                layer_names=layer_names,
                perturb_func=quantus.perturb_func.gaussian_noise,
                return_nan_when_prediction_changes=False,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="Relative Output Stability",
            category="Robustness",
            score_direction_lower_is_better=True,
            init=quantus.RelativeOutputStability(
                nr_samples=nr_samples,
                perturb_func=quantus.perturb_func.gaussian_noise,
                return_nan_when_prediction_changes=False,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="Relative Input Stability",
            category="Robustness",
            score_direction_lower_is_better=True,
            init=quantus.RelativeInputStability(
                nr_samples=nr_samples,
                perturb_func=quantus.perturb_func.gaussian_noise,
                return_nan_when_prediction_changes=False,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="Pixel-Flipping",
            category="Fidelity",
            score_direction_lower_is_better=True,
            init=quantus.PixelFlipping(
                features_in_step=features,
                perturb_baseline=perturb_baseline,
                perturb_func=perturb_func,
                return_auc_per_sample=True,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="Faithfulness Correlation",
            category="Fidelity",
            score_direction_lower_is_better=False,
            init=quantus.FaithfulnessCorrelation(
                subset_size=features,
                perturb_baseline=perturb_baseline,
                perturb_func=perturb_func,
                nr_runs=nr_samples,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="Faithfulness Estimate",
            category="Fidelity",
            score_direction_lower_is_better=True,
            init=quantus.FaithfulnessEstimate(
                features_in_step=features,
                perturb_baseline=perturb_baseline,
                perturb_func=perturb_func,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="Region Perturbation",
            category="Fidelity",
            score_direction_lower_is_better=True,
            init=quantus.RegionPerturbation(
                patch_size=patch_size,
                return_auc_per_sample=True,
                perturb_baseline=perturb_baseline,
                perturb_func=perturb_func,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="MPRT",
            category="Sensitivity",
            score_direction_lower_is_better=True,
            init=quantus.MPRT(
                skip_layers=True,
                return_last_correlation=True,
                similarity_func=similarity_func,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="sMPRT",
            category="Sensitivity",
            score_direction_lower_is_better=True,
            init=quantus.SmoothMPRT(
                return_last_correlation=True,
                skip_layers=True,
                nr_samples=nr_samples,
                noise_magnitude=x_noise,
                similarity_func=similarity_func,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="eMPRT",
            category="Sensitivity",
            score_direction_lower_is_better=False,
            init=quantus.EfficientMPRT(
                similarity_func=similarity_func,
                skip_layers=True,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="GEF",
            category="Unified",
            score_direction_lower_is_better=False,
            init=GEF(
                M=M,
                Z=Z,
                fast_mode=False,
                similarity_func=similarity_func,
                num_classes=num_classes,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                aggregate_func=np.mean,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
        Estimator(
            name="Fast-GEF",
            category="Unified",
            score_direction_lower_is_better=False,
            init=GEF(
                M=M,
                K=K,
                Z=Z,
                T=T,
                fast_mode=True,
                similarity_func=similarity_func,
                num_classes=num_classes,
                abs=abs,
                normalise=normalise,
                normalise_func=normalise_func,
                return_aggregate=return_aggregate,
                aggregate_func=np.mean,
                disable_warnings=disable_warnings,
                display_progressbar=display_progressbar,
            ),
        ),
    ]

    estimator_dict = {}
    for estimator in estimators:
        if estimator.category not in estimator_dict:
            estimator_dict[estimator.category] = {}
        estimator_dict[estimator.category][estimator.name] = {
            "init": estimator.init,
            "score_direction": estimator.score_direction_lower_is_better,
        }

    return estimator_dict

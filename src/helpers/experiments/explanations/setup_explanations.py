from typing import Dict, Optional, List
import numpy as np
import torch
import torchvision
from transformers import AutoModelForCausalLM, AutoTokenizer
from zennit import canonizers, composites, rules, attribution
from zennit import torchvision as zvision
from zennit import types as ztypes

from .utils_dv import register_final_layer_hook


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


def get_parameterised_explanations(
    task: str,
    xai_methods: List[str],
    device: torch.device,
    model: torch.nn.Module,
    am_batch: Optional[torch.Tensor],
    llm_explainer_name: str,
    tokenizer: AutoTokenizer,
    am_steps: int,
    nr_channels: int,
    img_size: Optional[int],
    class_labels: List[str],
    subtask: str,
    top_K: int,
    tokenizer_max_length: int,
    xai_layer_name: Optional[str] = None,
    is_top_K: bool = False,
):
    # try:
    methods = {
        "SmoothGrad": {
            "xai_lib": "zennit",
            "attributor": attribution.SmoothGrad,
            "attributor_kwargs": {"n_iter": 10, "noise_level": 0.1},
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
        },
        "IntegratedGradients": {
            "xai_lib": "zennit",
            "attributor": attribution.IntegratedGradients,
            "attributor_kwargs": {
                "n_iter": 10,
            },
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
        },
        "LRP-Eps": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "composite": Epsilon,
            "canonizer": get_zennit_canonizer(model),
            "canonizer_kwargs": {},
            "composite_kwargs": {"stabilizer": 1e-6, "epsilon": 1e-6},
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
        },
        "Guided-Backprop": {
            "xai_lib": "zennit",
            "attributor": attribution.Gradient,
            "canonizer": None,
            "composite": composites.GuidedBackprop,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
        },
        "LayerIntegratedGradients": {
            "xai_lib": "captum",
            "gc_layer": xai_layer_name,
            "xai_lib_attrib_kwargs": {"n_steps": 10},
        },
        "Gradient": {
            "xai_lib": "captum",
            "device": device,
        },
        "Saliency": {
            "xai_lib": "captum",
        },
        "DeepLift": {
            "xai_lib": "captum",
        },
        "GradientShap": {
            "xai_lib": "captum",
            "xai_lib_attrib_kwargs": {"n_samples": 10},
        },
        "InputXGradient": {
            "xai_lib": "captum",
        },
        "Control Var. Sobel Filter": {
            "xai_lib": "captum",
        },
        "Control Var. Random Uniform": {
            "xai_lib": "captum",
        },
        "Control Var. Constant": {
            "xai_lib": "captum",
            "constant_value": "black",
        },
        "Act-Max": {
            "am_steps": am_steps,
            "layer_name": "register_final_layer_hook(model, activation_dictionary)",
        },
        "LLM-x": {
            "top_K": top_K,
            "llm_name": llm_explainer_name,
            "class_labels": class_labels,
            "subtask": subtask,
            "attention_mask": am_batch,
            "tokenizer": tokenizer,
            "tokenizer_max_length": tokenizer_max_length,
        },
        "PartitionShap": {
            "tokenizer": tokenizer,
            "tokenizer_max_length": tokenizer_max_length,
        },
        "Random Guess": {},
        "Random Guess K": {
            "top_K": top_K,
        },
    }

    if img_size is not None:
        methods = {
            **methods,
            **{
                "LayerGradCam": {
                    "xai_lib": "captum",
                    "gc_layer": xai_layer_name,
                    "interpolate": (img_size, img_size),
                    "interpolate_mode": "bilinear",
                },
                "Occlusion": {
                    "xai_lib": "captum",
                    "window": (nr_channels, int(img_size / 4), int(img_size / 4)),
                },
                "MACO": {
                    "am_steps": am_steps,
                    "img_size": img_size,
                },
                "Fourier": {
                    "am_steps": am_steps,
                    "img_size": img_size,
                },
            },
        }

    xai_methods_with_kwargs = {}
    for xai in xai_methods:
        try:
            xai_methods_with_kwargs[xai] = methods.get(xai)
            if "K=" in xai:
                xai_methods_with_kwargs[xai] = {}
                xai_methods_with_kwargs[xai]["top_K"] = int(xai.split("K=")[1])
        except Exception as e:
            print(f"XAI method {xai} not found in methods dictionary.\n{e}")
            xai_methods_with_kwargs[xai] = {}

    for xai in xai_methods_with_kwargs:
        try:
            xai_methods_with_kwargs[xai]["task"] = task
            xai_methods_with_kwargs[xai]["method"] = xai
            xai_methods_with_kwargs[xai]["device"] = device

            if is_top_K:
                xai_methods_with_kwargs[xai]["is_top_K"] = is_top_K
                xai_methods_with_kwargs[xai]["top_K"] = top_K

            if "LLM" in xai:
                xai_methods_with_kwargs[xai]["llm_tokenizer"] = (
                    AutoTokenizer.from_pretrained(llm_explainer_name)
                )
                xai_methods_with_kwargs[xai]["llm_model"] = (
                    AutoModelForCausalLM.from_pretrained(llm_explainer_name)
                )

        except Exception as e:
            print(f"Error in setting up XAI method {xai}.\n{e}")

    ALT_TEXT = [
        "Gradient",
        "Saliency",
        "LayerIntegratedGradients",
        "PartitionShap",
        "IntegratedGradients",
        "Guided-Backprop",
        "GradientShap",
    ]

    for xai in xai_methods_with_kwargs:
        if task == "text" and xai in ALT_TEXT:
            xai_methods_with_kwargs[xai] = {
                "method": xai,
                "task": task,
                "device": device,
                "n_steps": 10,
                "n_samples": 10,
                "stdevs": 0.1,
                "tokenizer": tokenizer,
                "return_convergence_delta": False,
                "token_ids": "tokenizer.pad_token_id",
                "gc_layer": xai_layer_name,
                # "internal_batch_size": 10,
                "tokenizer_max_length": tokenizer_max_length,
            }

    if "LayerGradCam" in xai_methods_with_kwargs.keys() and xai_layer_name is None:
        del xai_methods_with_kwargs["LayerGradCam"]
    if (
        "LayerIntegratedGradients" in xai_methods_with_kwargs.keys()
        and xai_layer_name is None
    ):
        del xai_methods_with_kwargs["LayerIntegratedGradients"]

    if "Act-Max" in xai_methods_with_kwargs:
        # Automatically register hook to the final layer.
        # Check the modules.
        for module in enumerate(model.named_children()):
            print(module)
        print()
    if "PartitionShap" in xai_methods_with_kwargs and task != "text":
        del xai_methods_with_kwargs["PartitionShap"]

    return xai_methods_with_kwargs

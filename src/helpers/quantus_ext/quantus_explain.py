"""This modules contains customised explainer function which can be used with Quantus metrics to replace for quantus.explain."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import warnings
from importlib import util
from typing import Optional, Union, Callable

import numpy as np
import quantus
import torch
import scipy

from quantus.helpers import constants
from quantus.helpers import __EXTRAS__
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.utils import (
    get_baseline_value,
    infer_channel_first,
    get_wrapped_model,
)
import gc
import transformers
import shap


if util.find_spec("torch"):
    import torch
if util.find_spec("captum"):
    from captum.attr import (
        GradientShap,
        IntegratedGradients,
        InputXGradient,
        Saliency,
        Occlusion,
        FeatureAblation,
        LayerGradCam,
        GuidedBackprop,
        LayerIntegratedGradients,
        NeuronGradient,
        DeepLift,
        DeepLiftShap,
        GuidedGradCam,
        Deconvolution,
        FeaturePermutation,
        Lime,
        KernelShap,
        LRP,
        LayerConductance,
        LayerActivation,
        InternalInfluence,
        LayerGradientXActivation,
    )
if util.find_spec("zennit"):
    from zennit import canonizers as zcanon
    from zennit import composites as zcomp
    from zennit import attribution as zattr
    from zennit import core as zcore
if util.find_spec("tensorflow"):
    import tensorflow as tf
if util.find_spec("tf_explain"):
    import tf_explain

if util.find_spec("transformers"):
    from transformers import (
        PreTrainedModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed,
    )

from src.quantus_metric import get_wrapped_model_gef
from src.utils_act_max import (
    act_max,
    register_final_layer_hook,
)
from src.utils_autointerpret import *
from src.utils_maco import *


def explain_gef(model, inputs, targets, device, **kwargs) -> np.ndarray:
    """
    Explain inputs given a model, targets and an explanation method.
    Expecting inputs to be shaped such as (batch_size, nr_channels, ...) or (batch_size, ..., nr_channels).

    Parameters
    ----------
    model: torch.nn.Module, tf.keras.Model
            A model that is used for explanation.
    inputs: np.ndarray
             The inputs that ought to be explained.
    targets: np.ndarray
             The target lables that should be used in the explanation.
    kwargs: optional
            Keyword arguments. Pass as "explain_func_kwargs" dictionary when working with a metric class.
            Pass as regular kwargs when using the stnad-alone function.

            xai_lib: string, optional
                XAI library: captum, tf-explain or zennit.
            method: string, optional
                XAI method (used with captum and tf-explain libraries).
            attributor: string, optional
                XAI method (used with zennit).
            xai_lib_kwargs: dictionary, optional
                Keyword arguments to be passed to the attribution function.
            softmax: boolean, optional
                Indicated whether softmax activation in the last layer shall be removed.
            channel_first: boolean, optional
                Indicates if the image dimensions are channel first, or channel last.
                Inferred from the input shape if None.
            reduce_axes: tuple
                Indicates the indices of dimensions of the output explanation array to be summed. For example, an input
                array of shape (8, 28, 28, 3) with keepdims=True and reduce_axes = (-1,) will return an array of shape
                (8, 28, 28, -1). Passing "()" will keep the original dimensions.
            keepdims: boolean
                Indicated if the reduced axes shall be preserved (True) or removed (False).

    Returns
    -------
    explanation: np.ndarray
             Returns np.ndarray of same shape as inputs.
    """

    if util.find_spec("captum") or util.find_spec("tf_explain"):
        if "method" not in kwargs:
            warnings.warn(
                f"Using quantus 'explain' function as an explainer without specifying 'method' (string) "
                f"in kwargs will produce a vanilla 'Gradient' explanation.\n",
                category=UserWarning,
            )
    elif util.find_spec("zennit"):
        if "attributor" not in kwargs:
            warnings.warn(
                f"Using quantus 'explain' function as an explainer without specifying 'attributor'"
                f"in kwargs will produce a vanilla 'Gradient' explanation.\n",
                category=UserWarning,
            )

    elif not __EXTRAS__:
        raise ImportError(
            "Explanation library not found. Please install Captum or Zennit for torch>=1.2 models "
            "and tf-explain for TensorFlow>=2.0."
        )

    explanation = get_explanation(model, inputs, targets, device, **kwargs)

    # Convert the explanation batch (nr_samples, explanations) to a binary explanation vectors, with 1 where the top-k attributions are.
    if "is_top_K" in kwargs and "top_K" in kwargs:
        if kwargs["is_top_K"]:
            explanation = get_top_k_explanation(explanation, kwargs["top_K"])

    return explanation


def get_top_k_explanation(explanation: np.ndarray, top_K: int) -> np.ndarray:
    """
    Get the top-k attributions from an explanation array.

    Parameters
    ----------
    explanation: np.ndarray
        The explanation array.
    top_k: int
        The number of top-k attributions to keep.

    Returns
    -------
    explanation: np.ndarray
        Returns np.ndarray of same shape as inputs with binary values.
    """

    # Flatten the explanation array while keeping the batch dimension intact.
    batch_size = explanation.shape[0]
    flattened_explanation = explanation.reshape(batch_size, -1)

    # Create a binary explanation array.
    binary_explanation = np.zeros_like(flattened_explanation)

    # Get the top-k indices for each sample in the batch.
    for i in range(batch_size):
        top_k_indices = np.argsort(flattened_explanation[i])[-top_K:]
        binary_explanation[i, top_k_indices] = 1

    # Reshape the binary explanation array back to the original shape.
    binary_explanation = binary_explanation.reshape(explanation.shape)

    return binary_explanation


def get_explanation(model, inputs, targets, device, **kwargs):
    """
    Generate explanation array based on the type of input model and user specifications.
    For tensorflow models, tf.explain is used.
    For pytorch models, either captum or zennit is used, depending on which module is installed.
    If both are installed, captum is used per default. Setting the xai_lib kwarg to "zennit" uses zennit instead.

    Parameters
    ----------
    model: torch.nn.Module, tf.keras.Model
            A model that is used for explanation.
    inputs: np.ndarray
         The inputs that ought to be explained.
    targets: np.ndarray
         The target lables that should be used in the explanation.
    kwargs: optional
            Keyword arguments. Pass as "explain_func_kwargs" dictionary when working with a metric class.
            Pass as regular kwargs when using the stnad-alone function.

            xai_lib: string, optional
                XAI library: captum, tf-explain or zennit.
            method: string, optional
                XAI method (used with captum and tf-explain libraries).
            attributor: string, optional
                XAI method (used with zennit).
            xai_lib_kwargs: dictionary, optional
                Keyword arguments to be passed to the attribution function.
            softmax: boolean, optional
                Indicated whether softmax activation in the last layer shall be removed.
            channel_first: boolean, optional
                Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape if None.
            reduce_axes: tuple
                Indicates the indices of dimensions of the output explanation array to be summed. For example, an input
                array of shape (8, 28, 28, 3) with keepdims=True and reduce_axes = (-1,) will return an array of shape
                (8, 28, 28, -1). Passing "()" will keep the original dimensions.
            keepdims: boolean
                Indicated if the reduced axes shall be preserved (True) or removed (False).
    Returns
    -------
    explanation: np.ndarray
         Returns np.ndarray of same shape as inputs.
    """
    xai_lib = kwargs.get("xai_lib", "captum")
    task = kwargs.get("task")
    method = kwargs.get("method")
    functions = [
        # generate_captum_text_explanation,
        # generate_feature_visualisation_explanation,
        # generate_zennit_explanation,
        # generate_captum_explanation,
        # generate_llmx_explanation,
        # generate_random_explanation,
        # generate_shap_text_explanation,
    ]
    # try:
    # Specific explanation functions.
    if "LLM" in method:
        return generate_llmx_explanation(model, inputs, targets, device, **kwargs)
    elif "Control Var." in method:
        return generate_captum_explanation(model, inputs, targets, device, **kwargs)
    elif "Random Guess" in method:
        return generate_random_explanation(model, inputs, targets, device, **kwargs)

    elif task == "text":
        # print("task found as text in get_explanation")
        if "PartitionShap" == method:
            return generate_shap_text_explanation(
                model, inputs, targets, device, **kwargs
            )
        else:
            return generate_captum_text_explanation(
                model, inputs, targets, device, **kwargs
            )
    elif method in ["Act-Max", "MACO", "Fourier"]:
        return generate_feature_visualisation_explanation(
            model, inputs, targets, device, **kwargs
        )
    elif xai_lib == "zennit":
        return generate_zennit_explanation(model, inputs, targets, device, **kwargs)
    else:
        return generate_captum_explanation(model, inputs, targets, device, **kwargs)


def generate_captum_explanation(
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate explanation for a torch model with captum.
    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
         The inputs that ought to be explained.
    targets: np.ndarray
         The target lables that should be used in the explanation.
    device: string
        Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    kwargs: optional
            Keyword arguments. Pass as "explain_func_kwargs" dictionary when working with a metric class.
            Pass as regular kwargs when using the stnad-alone function. May include xai_lib_kwargs dictionary which includes keyword arguments for a method call.

            xai_lib: string
                XAI library: captum, tf-explain or zennit.
            method: string
                XAI method.
            xai_lib_kwargs: dict
                Keyword arguments to be passed to the attribution init function.
            xai_lib_attrib_kwargs: dict
                Keyword arguments to be passed to the attribute call function.
            channel_first: boolean, optional
                Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape if None.
            reduce_axes: tuple
                Indicates the indices of dimensions of the output explanation array to be summed. For example, an input
                array of shape (8, 28, 28, 3) with keepdims=True and reduce_axes = (-1,) will return an array of shape
                (8, 28, 28, -1). Passing "()" will keep the original dimensions.
            keepdims: boolean
                Indicated if the reduced axes shall be preserved (True) or removed (False).
    Returns
    -------
    explanation: np.ndarray
         Returns np.ndarray of same shape as inputs.
    """

    device = device if device else kwargs.get("device", "cuda")

    channel_first = (
        kwargs["channel_first"]
        if "channel_first" in kwargs
        else infer_channel_first(inputs)
    )

    softmax = kwargs.get("softmax", None)
    if softmax is not None:
        warnings.warn(
            f"Softmax argument has been passed to the explanation function. Different XAI "
            f"methods may or may not require the output to go through softmax activation. "
            f"Make sure that your softmax argument choice aligns with the method intended usage.\n",
            category=UserWarning,
        )
        wrapped_model = get_wrapped_model(
            model, softmax=softmax, channel_first=channel_first
        )
        model = wrapped_model.get_softmax_arg_model()

    method = kwargs.get("method", "Gradient")
    xai_lib_kwargs = kwargs.get("xai_lib_kwargs", {})
    xai_lib_attrib_kwargs = kwargs.get("xai_lib_attrib_kwargs", {})

    # Set model in evaluate mode.
    model.to(device)
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)  # , dtype=torch.float64
        # print("updated")

    # Update inputs to torch.float64
    inputs = inputs.float()
    inputs.requires_grad = True

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, dtype=int).to(device)

    assert 0 not in kwargs.get(
        "reduce_axes", [1]
    ), "Reduction over batch_axis is not available, please do not include axis 0 in 'reduce_axes' kwargs."
    assert len(kwargs.get("reduce_axes", [1])) <= inputs.ndim - 1, (
        "Cannot reduce attributions over more axes than each sample has dimensions, but got "
        "{} and  {}.".format(len(kwargs.get("reduce_axes", [1])), inputs.ndim - 1)
    )

    reduce_axes = {
        "axis": tuple(kwargs.get("reduce_axes", [1])),
        "keepdims": kwargs.get("keepdims", True),
    }

    # Prevent attribution summation for 2D-data. Recreate np.sum behavior when passing reduce_axes=(), i.e. no change.
    if (len(tuple(kwargs.get("reduce_axes", [1]))) == 0) | (inputs.ndim < 3):

        def f_reduce_axes(a):
            return a

    else:

        def f_reduce_axes(a):
            return a.sum(**reduce_axes)

    explanation: torch.Tensor = torch.zeros_like(inputs)

    if method in constants.DEPRECATED_XAI_METHODS_CAPTUM:
        warnings.warn(
            f"Explanaiton method string {method} is deprecated. Use "
            f"{constants.DEPRECATED_XAI_METHODS_CAPTUM[method]} instead.\n",
            category=UserWarning,
        )
        method = constants.DEPRECATED_XAI_METHODS_CAPTUM[method]

    elif method in ["GradientShap", "DeepLift", "DeepLiftShap"]:
        baselines = (
            kwargs["baseline"] if "baseline" in kwargs else torch.zeros_like(inputs)
        )
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs,
                target=targets,
                baselines=baselines,
                **xai_lib_attrib_kwargs,
            )
        )

    elif method == "IntegratedGradients":
        baselines = (
            kwargs["baseline"] if "baseline" in kwargs else torch.zeros_like(inputs)
        )
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs,
                target=targets,
                baselines=baselines,
                method="riemann_trapezoid",
                **xai_lib_attrib_kwargs,
            )
        )

    elif method in [
        "InputXGradient",
        "Saliency",
        "FeatureAblation",
        "Deconvolution",
        "FeaturePermutation",
        "Lime",
        "KernelShap",
        "LRP",
    ]:
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs, target=targets, **xai_lib_attrib_kwargs
            )
        )

    elif method == "Gradient":
        explanation = f_reduce_axes(
            Saliency(model, **xai_lib_kwargs).attribute(
                inputs=inputs, target=targets, abs=False, **xai_lib_attrib_kwargs
            )
        )

    elif method == "Occlusion":
        window_shape = kwargs.get("window", (1, *([4] * (inputs.ndim - 2))))
        explanation = f_reduce_axes(
            Occlusion(model).attribute(
                inputs=inputs,
                target=targets,
                sliding_window_shapes=window_shape,
                **xai_lib_attrib_kwargs,
            )
        )

    elif method in [
        "LayerGradCam",
        "GuidedGradCam",
        "LayerConductance",
        "LayerActivation",
        "InternalInfluence",
        "LayerGradientXActivation",
        "LayerIntegratedGradients",
    ]:
        if "gc_layer" in kwargs:
            xai_lib_kwargs["layer"] = kwargs["gc_layer"]

        if "layer" not in xai_lib_kwargs:
            raise ValueError(
                "Specify a convolutional layer name as 'gc_layer' to run GradCam."
            )

        if isinstance(xai_lib_kwargs["layer"], str):
            xai_lib_kwargs["layer"] = eval(xai_lib_kwargs["layer"])

        attr_func = eval(method)

        if method != "LayerActivation":
            explanation = attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs, target=targets, **xai_lib_attrib_kwargs
            )
        else:
            explanation = attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs, **xai_lib_attrib_kwargs
            )

        if "interpolate" in kwargs:
            if isinstance(kwargs["interpolate"], tuple):
                if "interpolate_mode" in kwargs:
                    explanation = LayerGradCam.interpolate(
                        explanation,
                        kwargs["interpolate"],
                        interpolate_mode=kwargs["interpolate_mode"],
                    )
                else:
                    explanation = LayerGradCam.interpolate(
                        explanation, kwargs["interpolate"]
                    )
        else:
            if explanation.shape[-1] != inputs.shape[-1]:
                warnings.warn(
                    "Quantus requires GradCam attribution and input to correspond in "
                    "last dimensions, but got shapes {} and {}\n "
                    "Pass 'interpolate' argument to explanation function get matching dimensions.".format(
                        explanation.shape, inputs.shape
                    ),
                    category=UserWarning,
                )

        explanation = f_reduce_axes(explanation)

    elif method == "Control Var. Sobel Filter":
        explanation = torch.zeros(size=inputs.shape)

        if inputs.is_cuda:
            inputs = inputs.cpu()

        inputs_numpy = inputs.detach().numpy()

        for i in range(len(explanation)):
            explanation[i] = torch.Tensor(
                np.clip(scipy.ndimage.sobel(inputs_numpy[i]), 0, 1)
            )
        if len(explanation.shape) > 2:
            explanation = explanation.mean(**reduce_axes)

    elif method == "Control Var. Random Uniform":
        explanation = torch.rand(size=(inputs.shape))
        if len(explanation.shape) > 2:
            explanation = explanation.mean(**reduce_axes)

    elif method == "Control Var. Constant":
        assert (
            "constant_value" in kwargs
        ), "Specify a 'constant_value' e.g., 0.0 or 'black' for pixel replacement."

        explanation = torch.zeros(size=inputs.shape)

        # Update the tensor with values per input x.
        for i in range(explanation.shape[0]):
            constant_value = get_baseline_value(
                value=kwargs["constant_value"], arr=inputs[i], return_shape=(1,)
            )[0]
            explanation[i] = torch.Tensor().new_full(
                size=explanation[0].shape, fill_value=constant_value
            )

        if len(explanation.shape) > 2:
            explanation = explanation.mean(**reduce_axes)

    else:
        raise KeyError(
            f"The selected {method} XAI method is not in the list of supported built-in Quantus XAI methods for Captum. "
            f"Please choose an XAI method that has already been implemented {constants.AVAILABLE_XAI_METHODS_CAPTUM}."
        )

    # Sum over the axes for image data.
    if len(explanation.shape) > 2:
        explanation = torch.mean(explanation, **reduce_axes)

    if kwargs.get("keep_on_tensors", False):
        return explanation

    elif isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    return explanation


def generate_zennit_explanation(
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate explanation for a torch model with zennit.

    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
         The inputs that ought to be explained.
    targets: np.ndarray
         The target lables that should be used in the explanation.
    device: string
        Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    kwargs: optional
            Keyword arguments. Pass as "explain_func_kwargs" dictionary when working with a metric class.
            Pass as regular kwargs when using the stnad-alone function.

            attributor: string, optional
                XAI method.
            xai_lib_kwargs: dictionary, optional
                Keyword arguments to be passed to the attribution function.
            softmax: boolean, optional
                Indicated whether softmax activation in the last layer shall be removed.
            channel_first: boolean, optional
                Indicates if the image dimensions are channel first, or channel last.
                Inferred from the input shape if None.
            reduce_axes: tuple
                Indicates the indices of dimensions of the output explanation array to be summed. For example, an input
                array of shape (8, 28, 28, 3) with keepdims=True and reduce_axes = (-1,) will return an array of shape
                (8, 28, 28, -1). Passing "()" will keep the original dimensions.
            keepdims: boolean
                Indicated if the reduced axes shall be preserved (True) or removed (False).
    Returns
    -------
    explanation: np.ndarray
         Returns np.ndarray of same shape as inputs.

    """

    device = device if device else kwargs.get("device", "cuda")

    channel_first = (
        kwargs["channel_first"]
        if "channel_first" in kwargs
        else infer_channel_first(inputs)
    )
    softmax = kwargs.get("softmax", None)

    if softmax is not None:
        warnings.warn(
            f"Softmax argument has been passed to the explanation function. Different XAI "
            f"methods may or may not require the output to go through softmax activation. "
            f"Make sure that your softmax argument choice aligns with the method intended usage.\n",
            category=UserWarning,
        )
        wrapped_model = get_wrapped_model_gef(
            model, softmax=softmax, channel_first=channel_first
        )
        model = wrapped_model.get_softmax_arg_model()

    assert 0 not in kwargs.get(
        "reduce_axes", [1]
    ), "Reduction over batch_axis is not available, please do not include axis 0 in 'reduce_axes' kwarg."
    assert len(kwargs.get("reduce_axes", [1])) <= inputs.ndim - 1, (
        "Cannot reduce attributions over more axes than each sample has dimensions, but got "
        "{} and  {}.".format(len(kwargs.get("reduce_axes", [1])), inputs.ndim - 1)
    )

    reduce_axes = {
        "axis": tuple(kwargs.get("reduce_axes", [1])),
        "keepdims": kwargs.get("keepdims", True),
    }

    # Get zennit composite, canonizer, attributor and handle canonizer kwargs.
    canonizer = kwargs.get("canonizer", None)
    if not canonizer == None and not issubclass(canonizer, zcanon.Canonizer):
        raise ValueError(
            "The specified canonizer is not valid. "
            "Please provide None or an instance of zennit.canonizers.Canonizer"
        )

    # Handle attributor kwargs.
    attributor = kwargs.get("attributor", zattr.Gradient)
    if not issubclass(attributor, zattr.Attributor):
        raise ValueError(
            "The specified attributor is not valid. "
            "Please provide a subclass of zennit.attributon.Attributor"
        )

    # Handle attributor kwargs.
    composite = kwargs.get("composite", None)
    if not composite == None and isinstance(composite, str):
        if composite not in zcomp.COMPOSITES.keys():
            raise ValueError(
                "Composite {} does not exist in zennit."
                "Please provide None, a subclass of zennit.core.Composite, or one of {}".format(
                    composite, zcomp.COMPOSITES.keys()
                )
            )
        else:
            composite = zcomp.COMPOSITES[composite]
    if not composite == None and not issubclass(composite, zcore.Composite):
        raise ValueError(
            "The specified composite {} is not valid. "
            "Please provide None, a subclass of zennit.core.Composite, or one of {}".format(
                composite, zcomp.COMPOSITES.keys()
            )
        )

    # Set model in evaluate mode.
    # model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(device)

    inputs.to(device)
    targets.to(device)

    canonizer_kwargs = kwargs.get("canonizer_kwargs", {})
    composite_kwargs = kwargs.get("composite_kwargs", {})
    attributor_kwargs = kwargs.get("attributor_kwargs", {})

    # Initialize canonizer, composite, and attributor.
    if canonizer is not None:
        canonizers = [canonizer(**canonizer_kwargs)]
    else:
        canonizers = []
    if composite is not None:
        composite = composite(
            **{
                **composite_kwargs,
                "canonizers": canonizers,
            }
        )
    attributor = attributor(
        **{
            **attributor_kwargs,
            "model": model,
            "composite": composite,
        }
    )

    n_outputs = model(inputs).shape[1]

    # Get the attributions.
    with attributor:
        if "attr_output" in attributor_kwargs.keys():
            _, explanation = attributor(inputs, None)
        else:
            eye = torch.eye(n_outputs, device=device)
            output_target = eye[targets]
            output_target = output_target.reshape(-1, n_outputs)
            _, explanation = attributor(inputs, output_target)

    # print(f"explamation shape inside function {explanation.shape})")
    # Sum over the axes for image data.
    if len(explanation.shape) > 2:
        explanation = torch.sum(explanation, **reduce_axes)
    # print(f"explamation shape inside function2 {explanation.shape})")
    if kwargs.get("keep_on_tensors", False):
        return explanation

    elif isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    return explanation


def generate_feature_visualisation_explanation(
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:

    device = device if device else kwargs.get("device", "cuda")
    method = kwargs.get("method", "Act-Max")
    am_steps = kwargs["am_steps"]
    img_size = kwargs.get("img_size", 224)

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(device)

    # Assumes batch like (1, 3, 227, 227), otherwise use .unsqueeze(0).
    inputs_random = torch.rand_like(inputs.to(torch.float)).to(device)
    inputs_random.requires_grad_(True)

    explanation: torch.Tensor = torch.zeros_like(inputs_random)
    model.eval().to(device)

    # Generate explanations.
    for i in range(len(inputs_random)):

        if method == "Act-Max":

            activation_dictionary = {}
            layer_name = eval(kwargs.get("layer_name"))
            alpha = kwargs.get("alpha", torch.tensor(50))

            explanation[i] = act_max(
                network=model,
                input_img=inputs_random[i],
                layer_activation=activation_dictionary,
                layer_name=layer_name,
                unit=targets[i],
                steps=am_steps,
                alpha=alpha,
                device=device,
            )
        else:
            x = inputs_random[i].unsqueeze(0)
            objective = lambda x: torch.mean(model(x)[:, targets[i]])

            if method == "MACO":
                # second dim is "alpha".
                explanation[i] = maco(
                    objective,
                    total_steps=am_steps,
                    learning_rate=1.0,
                    image_size=img_size,
                    model_input_size=img_size,
                    noise=0.1,
                    values_range=(-2.5, 2.5),
                    crops_per_iteration=6,
                    box_size=(0.20, 0.25),
                    device=device,
                    nr_channels=inputs.shape[1],
                )[0]

            elif method == "Fourier":

                explanation[i] = fourier(
                    objective,
                    total_steps=am_steps,
                    decay_power=1.5,
                    learning_rate=1.0,
                    image_size=img_size,
                    model_input_size=img_size,
                    noise=0.1,
                    values_range=(-2.5, 2.5),
                    crops_per_iteration=6,
                    box_size=(0.20, 0.25),
                    device=device,
                    nr_channels=inputs.shape[1],
                )[0]

    # Sum over the axes for image data, and if the second dimenion (nr_channels) has three channels, sum over the second dimension.
    if len(explanation.shape) == 4 and explanation.shape[1] == 3:
        explanation = explanation.mean(axis=1)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    return explanation


def generate_llmx_explanation(
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate explanation for a torch model with llmx.

    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
        The inputs that ought to be explained.
    targets: np.ndarray
        The target lables that should be used in the explanation.
    device: string
        Device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    """

    device = device if device else kwargs.get("device", "cuda")
    torch.cuda.empty_cache()

    tokenizer = kwargs["tokenizer"]
    subtask = kwargs["subtask"]
    class_labels = kwargs["class_labels"]
    top_K = kwargs["top_K"]
    verbose = kwargs.get("verbose", False)

    # Initialise the language model and tokeniser.
    llm_tokenizer = kwargs["llm_tokenizer"]
    llm_model = kwargs["llm_model"]
    llm_model.to(device)

    # Get the input IDs, attention masks and targets in the correct format.
    input_ids = torch.Tensor(inputs).to(device).to(torch.long)
    input_text = [
        tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids
    ]
    targets = torch.Tensor(targets).to(device).to(torch.long)
    # attention_mask = torch.Tensor(attention_mask).to(device).to(torch.long)

    softmaxs = torch.nn.functional.softmax(
        torch.Tensor(kwargs["logits_original"]), dim=1
    )
    softmaxs_perturb = torch.nn.functional.softmax(
        torch.Tensor(kwargs["logits_perturb"]), dim=1
    )

    max_length = inputs.shape[1]

    # Use LLM tokenizer to get the input ids from the original input text.
    input_ids = llm_tokenizer(
        input_text,
        padding="max_length",
        truncation=False,
        return_tensors="pt",
        max_length=max_length,  # tokenizer.model_max_length,  # inputs.shape[1]
    ).input_ids.to(device)

    # Convert input IDs back to text for the prompt.
    inputs_text_llm = [
        llm_tokenizer.decode(input_id, skip_special_tokens=True)
        for input_id in input_ids
    ]

    inputs = []
    prompts = []
    for index in range(len(targets)):
        # K = np.random.randint(1, 10) # FIXME.
        target = targets[index].item()

        # Rewrite input to LLM prompt.
        prompt = prepare_prompt(
            inputs=inputs_text_llm[index],
            target=target,
            softmax=softmaxs[index][target].detach().cpu().numpy(),
            softmax_perturb=softmaxs_perturb[index][target].detach().cpu().numpy(),
            subtask=subtask,
            class_labels=list(class_labels.values()),
            top_K=top_K,
            magnitude_level=(
                kwargs["magnitude_level"] if "magnitude_level" in kwargs else "None"
            ),  # moderately or significantly
        )
        prompts.append(prompt)
        inputs.append(inputs_text_llm[index])

    # Step 1B. Tokenize prompts, generate LLM responses, decode and parse.
    if verbose:
        print(f"Explaining with LLM: {kwargs['llm_name']}....")

    with torch.no_grad():
        prompt_tokens = llm_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        # Read more here. https://huggingface.co/blog/how-to-generate
        outputs = llm_model.generate(
            **prompt_tokens,
            max_length=tokenizer.model_max_length,
            num_return_sequences=1,  # TODO. Temperature description.
            temperature=0,
            do_sample=False,
        )
        decoded_outputs = [
            llm_tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        parsed_outputs = [
            safe_llm_parse(decoded_output, verbose=verbose)
            for decoded_output in decoded_outputs
        ]

        # Step 1A. Remove instances without a formatted repsponse.
        parsed_outputs = [{} if x is None else x for x in parsed_outputs]

        # Count the number of empty responses.
        empty_responses = sum([1 for x in parsed_outputs if x == {}])
        if verbose:
            print(f"{empty_responses} empty responses found given {len(inputs)}.")

        # Step 2. Generate explanations from tokens (Option 1. Binarise, then apply L2 or cosine. Option 2. Compare emedding representations. #eval("model."+experiment.xai_layer_name))
        explanations = generate_explanation_from_tokens(
            parsed_outputs,
            inputs,
            llm_tokenizer,
            max_length,
            binarise=True,
            verbose=verbose,
        )
        explanations = np.array([e.detach().cpu().numpy() for e in explanations])

        torch.cuda.empty_cache()

    return explanations


def generate_shap_text_explanation(model, inputs, targets, device, **kwargs):
    """
    Generate explanation for a torch model with SHAP.

    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
        The inputs that ought to be explained.
    targets: np.ndarray
        The target lables that should be used in the explanation.
    device: string
        Device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    """

    device = device if device else kwargs.get("device", "cuda")

    torch.cuda.empty_cache()
    tokenizer = kwargs["tokenizer"]
    tokenizer_max_length = kwargs["tokenizer_max_length"]

    # Get the input IDs, attention masks and targets in the correct format.
    input_ids = torch.Tensor(inputs).to(device).to(torch.long)
    input_text = [
        tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids
    ]

    # Build a pipeline object to do predictions.
    pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
    )

    explainer = shap.Explainer(pred)
    shap_values = explainer(input_text)

    # Initialize SHAP values matrix with shape (nr_samples, tokenizer_max_length).
    explanations = np.zeros((len(inputs), tokenizer_max_length))

    # Map SHAP values back to input IDs using offset mapping.
    for i, text in enumerate(input_text):
        # Tokenize the text to get input IDs and offset mapping.
        tokenized_text = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=tokenizer_max_length,
        )
        # token_ids = tokenized_text["input_ids"]
        offsets = tokenized_text["offset_mapping"]
        target_idx = targets[i]

        # Distribute SHAP values according to the offset mapping
        for j, (start, end) in enumerate(offsets):
            if j < len(shap_values[i].values):
                shap_value = shap_values.values[i][j][target_idx]
                for k in range(start, end):
                    if k < tokenizer_max_length:
                        explanations[i, k] += shap_value

    return explanations


def generate_captum_text_explanation(model, inputs, targets, device, **kwargs):
    """
    Generate explanation for a torch model with captum.

    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
        The inputs that ought to be explained.
    targets: np.ndarray
        The target lables that should be used in the explanation.
    device: string
        Device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    """

    torch.cuda.empty_cache()

    def predict(input_ids, model, attention_mask, **kwargs):

        model.eval()  # original pytorch bert model.
        outputs = model(
            input_ids,
            **{"attention_mask": attention_mask.to(torch.long)},
        )
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        return outputs

    device = device if device else kwargs.get("device", "cuda")
    method = kwargs.get("method", "Gradient")

    xai_lib_attrib_kwargs = kwargs.get("xai_lib_xai_lib_attrib_kwargskwargs", {})
    tokenizer = kwargs.get("tokenizer")
    token_ids = kwargs.get("token_ids")
    attention_mask = kwargs.get("attention_mask")

    input_ids = torch.Tensor(inputs).to(device).to(torch.long)
    attention_mask = torch.Tensor(attention_mask).to(device).to(torch.long)
    targets = torch.Tensor(targets).to(device).to(torch.int64)

    model.eval()
    explanation: torch.Tensor = torch.zeros_like(input_ids)

    reduce_axes = {
        "axis": tuple(kwargs.get("reduce_axes", [1])),
        "keepdims": kwargs.get("keepdims", True),
    }

    # Define the forward function.
    funcs = {"forward_func": predict}

    if method == "Guided-Backprop":
        explanation = GuidedBackprop(model).attribute(
            inputs=input_ids,
            target=targets,
            additional_forward_args=attention_mask,  # {"attention_mask": attention_mask},
        )
        print("Guided-Backprop explanation!", explanation.shape)

    elif method in ["IntegratedGradients", "LayerIntegratedGradients", "GradientShap"]:

        # Construct reference token ids.
        ref_tokenized = torch.zeros_like(input_ids).long()
        ref_tokenized[:] = eval(token_ids)

        if method == "IntegratedGradients":

            method_func = IntegratedGradients
            xai_lib_attrib_kwargs["n_steps"] = kwargs.get("n_steps", 5)
            xai_lib_attrib_kwargs["internal_batch_size"] = kwargs.get(
                "internal_batch_size", None
            )
        elif method == "GradientShap":
            method_func = GradientShap
            xai_lib_attrib_kwargs["stdevs"] = kwargs.get("stdevs", 0.1)
            xai_lib_attrib_kwargs["n_samples"] = kwargs.get("n_samples", 5)

        elif method == "LayerIntegratedGradients":

            # Instead of using eval.
            def get_underlying_model(model):
                """
                Checks if the model is an instance of a wrapper class and returns the underlying model.
                Otherwise, returns the model itself. To help get the correct layer for LayerIntegratedGradients.
                """
                if hasattr(model, "is_additive"):
                    return model.original_model
                else:
                    # Assuming the model is not wrapped or does not need unwrapping.
                    return model

            # Use the lambda function with the original model
            model_to_use = get_underlying_model(model)

            try:
                gc_layer_str = kwargs.get("gc_layer")
                funcs["layer"] = eval("model_to_use" + "." + gc_layer_str)
            except:
                gc_layer_str = "".join(kwargs.get("gc_layer").split("model")[1:])
                funcs["layer"] = eval("model_to_use" + "." + gc_layer_str)

            # print(gc_layer_str)
            method_func = LayerIntegratedGradients
            xai_lib_attrib_kwargs["n_steps"] = kwargs.get("n_steps", 5)
            xai_lib_attrib_kwargs["internal_batch_size"] = kwargs.get(
                "internal_batch_size", None
            )

        # Explain how each feature of each token contributes to the model's prediction.
        explanation = method_func(**funcs).attribute(
            inputs=input_ids,
            baselines=ref_tokenized,
            target=targets,
            additional_forward_args=(
                model,
                attention_mask,
            ),
            # predict # TODO. Maybe wrong.
            **xai_lib_attrib_kwargs,
        )
        # print("explanation.shape", explanation.shape, method)
        # Sum over the layers.
        if len(explanation.shape) > 2:
            explanation = explanation.sum(dim=2)
        else:
            print("ValueError: Method not supported. Please use a supported method.")

    elif method == "Control Var. Sobel Filter":
        explanation = torch.zeros(size=inputs.shape)

        if inputs.is_cuda:
            inputs = inputs.cpu()

        inputs_numpy = inputs.detach().numpy()

        for i in range(len(explanation)):
            explanation[i] = torch.Tensor(
                np.clip(scipy.ndimage.sobel(inputs_numpy[i]), 0, 1)
            )
        if len(explanation.shape) > 2:
            explanation = explanation.mean(**reduce_axes)

    elif method == "Control Var. Random Uniform":
        explanation = torch.rand(size=(inputs.shape))
        if len(explanation.shape) > 2:
            explanation = explanation.mean(**reduce_axes)

    elif method == "Control Var. Constant":
        assert (
            "constant_value" in kwargs
        ), "Specify a 'constant_value' e.g., 0.0 or 'black' for pixel replacement."

        explanation = torch.zeros(size=inputs.shape)

        # Update the tensor with values per input x.
        for i in range(explanation.shape[0]):
            constant_value = get_baseline_value(
                value=kwargs["constant_value"], arr=inputs[i], return_shape=(1,)
            )[0]
            explanation[i] = torch.Tensor().new_full(
                size=explanation[0].shape, fill_value=constant_value
            )

        if len(explanation.shape) > 2:
            explanation = explanation.mean(**reduce_axes)

    gc.collect()
    torch.cuda.empty_cache()

    if kwargs.get("keep_on_tensors", False):
        return explanation

    elif isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    return explanation


def generate_random_explanation(model, inputs, targets, device=None, **kwargs):
    """
    Generate explanation for a torch model with random values.

    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
        The inputs that ought to be explained.
    targets: np.ndarray
        The target lables that should be used in the explanation.
    device: string
        Device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    """

    device = device if device else kwargs.get("device", "cuda")

    task = kwargs.get("task")

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)

    if task == "vision":
        if inputs.shape[1] in [1, 3]:  # Check for 1 or 3 channels
            shape = inputs.shape
        else:
            raise ValueError("Unsupported number of channels for vision data")
    else:
        shape = inputs.shape

    # If K in kwargs, return an explanation of zeros and randomly placed ones in the input.
    if "top_K" in kwargs:
        top_K = kwargs["top_K"]
        explanation = torch.zeros(size=shape)
        flat_shape = np.prod(shape[1:])  # Flattened shape except the batch size
        indices_ones = np.random.randint(0, flat_shape, size=top_K)
        explanation = explanation.view(shape[0], -1)  # Flatten the tensor for indexing
        explanation[:, indices_ones] = 1
        explanation = explanation.view(shape)  # Reshape back to original shape
    else:
        explanation = torch.rand(size=shape)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    return explanation

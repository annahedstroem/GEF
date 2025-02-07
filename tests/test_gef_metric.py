from typing import Union, Dict
from pytest_lazyfixture import lazy_fixture
import pytest
import numpy as np
import torch

# from quantus.functions.explanation_func import explain
from quantus.helpers.model.model_interface import ModelInterface

from ..src.explain import explain_gef as explain
from ..src.gef import GEF

import warnings

warnings.filterwarnings("ignore", category=Warning)  # DeprecationWarning # FIXME.

"""
pip install --upgrade tensorflow
pip install --upgrade numpy
pip install --upgrade tensorboard
pip install --upgrade flatbuffers
"""


@pytest.mark.gef
@pytest.mark.parametrize(
    "test_name,model,data,params,expected",
    [
        (
            "titantic",
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "num_classes": 2,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "GradientShap P = 5",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 5,
                    "M": 2,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "GradientShap Fast-GEF P = 5",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 5,
                    "M": 2,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "batch_size": 10,
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "1d 3ch conv test Random",
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Random Uniform",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Random titantic",
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "P": 10,
                    "M": 1,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Random Uniform",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Sobel titantic",
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "P": 10,
                    "M": 1,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Sobel Filter",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Constant titantic",
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "P": 10,
                    "M": 1,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Constant",
                        "constant_value": "black",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "GradientShap P = 2",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 2,
                    "M": 2,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "GradientShap P = 10",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 10,
                    "M": 2,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Saliency P = 5",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 5,
                    "M": 2,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Random",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 10,
                    "M": 2,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Random Uniform",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Grad input",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "input_mode": True,
                    "P": 5,
                    "M": 2,
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Sobel approx",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 10,
                    "M": 1,
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Sobel Filter",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Constant approx",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 10,
                    "M": 1,
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Constant",
                        "constant_value": "black",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "normal Input mode",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "input_mode": True,
                    "M": 10,
                    "num_classes": 10,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "Input mode p levels set",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "input_mode": True,
                    "num_classes": 10,
                    "p_levels": [0.1, 0.5, 1.0, 1.5],
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "1d 3ch conv test Input mode",
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "input_mode": True,
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "exact mode Grad",
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "P": 5,
                    "M": 5,
                    "T": 5,
                    "num_classes": 10,
                    "fast_mode": False,
                    "return_aggregate": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "1d 3ch conv test Sal",
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "1d 3ch conv test Sobel",
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Sobel Filter",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            "1d 3ch conv test Constant",
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "num_classes": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Constant",
                        "constant_value": "black",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
    ],
)
def test_gef(
    test_name: str,
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    print("\n#########################")
    print(f"test - {test_name}!")
    print("#########################")
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    # Load metric.
    metric = GEF(**init_params)
    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        custom_batch=None,
        **call_params,
    )

    print(
        f"\n{explain_func_kwargs['method']} - scores: {np.mean(scores):.3f}, (±{np.std(scores):.3f}) -\n{np.round(np.array(scores), 2)}"
    )
    if np.isnan(metric.bridge_scores).any():
        print("\n\tNaNs: bridge_scores", np.sum(np.isnan(metric.bridge_scores)))
        print("\n\tNaNs: scores", np.sum(np.isnan(scores)))

    # If either input array contains NaN values.
    if (
        np.count_nonzero(np.isnan(metric.distortion_e)) > 0
        or np.count_nonzero(np.isnan(metric.distortion_f)) > 0
    ):
        print(
            f"NaNs: distortion_e: {np.count_nonzero(np.isnan(metric.distortion_e))}, "
            f"d_f: {np.count_nonzero(np.isnan(metric.distortion_f))}"
        )

    for m_ix in range(metric.M):
        for s_ix in range(metric.batch_size):
            if np.allclose(
                metric.distortion_f[m_ix, :, s_ix],
                metric.distortion_e[m_ix, :, s_ix],
            ):
                print(
                    f"Both input arrays have the same values: "
                    f"{metric.distortion_f[m_ix, :, s_ix] == metric.distortion_e[m_ix, :, s_ix]}"
                )

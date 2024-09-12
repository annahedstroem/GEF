"""This module contains the implementation of the GEF metric for push to Quantus."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.


import gc
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from importlib import util
import numpy as np
from tqdm import tqdm
import random

from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers import utils
from quantus.functions.normalise_func import (
    normalise_by_average_second_moment_estimate,
)
from quantus.functions.similarity_func import correlation_spearman, distance_euclidean
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)

from src.quantus_ext.quantus_model_interface import ModelInterfaceGEF
from src.quantus_ext.quantus_metric import (
    MetricGEF,
    get_wrapped_model_gef,
)
from src.quantus_ext.quantus_explain import explain_gef

from typing import final

if util.find_spec("torch"):
    import torch


@final
class GEF(MetricGEF):
    """
    Implementation of the Generalised Explanation Faithfulness (GEF) metric by anonymous et al., 2024.

    INSERT DESC.

    References:
        1) INSERT CITATION.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "GEF"
    data_applicability = {
        DataType.IMAGE,
        DataType.TIMESERIES,
        DataType.TABULAR,
        DataType.TEXT,
    }
    model_applicability = {ModelType.TORCH}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        similarity_func: Optional[Callable] = None,
        model_disortion_func: Optional[Callable] = None,
        explanation_disortion_func: Optional[Callable] = None,
        evaluate_class_distortion: bool = True,
        Z: int = 5,
        M: int = 5,
        perturbation_path: Optional[Union[np.ndarray, torch.Tensor]] = None,
        fast_mode: bool = False,
        pullback_kwargs: Optional[Dict[str, Any]] = None,
        activation_noise: Optional[float] = 1e-4,
        K: Optional[int] = 10,
        T: Optional[int] = 10,
        num_classes: Optional[int] = None,
        layer_idx: Optional[str] = None,
        input_mode: bool = False,
        noise_type: str = "multiplicative",
        mean: float = 1.0,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = True,
        display_progressbar: bool = False,
        debug: bool = False,
        base_seed: Optional[int] = 42,
        **kwargs,
    ):
        """
        Initialises the GEF metric with all relevant attributes.

        Parameters
        ----------
        similarity_func: callable, optional
            The similarity metric to use for comparing distortions.
        model_disortion_func: callable, optional
            The distance metric to use for comparing model outputs.
        explanation_disortion_func: callable, optional
            The distance metric to use for comparing explanations.
        evaluate_class_distortion: bool, optional
            Whether to evaluate class-specific distortions.
        Z: int, optional
            The number of perturbation levels.
        M: int, optional
            The number of iterations.
        perturbation_path: np.ndarray, optional
            The perturbation path to use.
        num_classes: int, optional
            The number of classes.
        layer_idx: str, optional
            The layer index.
        input_mode: bool, optional
            Whether to use input mode.
        noise_type: str, optional
            The type of noise to use.
        mean: float, optional
            The mean.
        fast_mode: bool, optional
            Whether to use fast mode.
        pullback_kwargs: dict, optional
            The pullback keyword arguments.
        activation_noise: float, optional
            The activation noise.
        K: int, optional
            The number of magnitude levels.
        T: int, optional
            The number of iterations.
        abs: bool, optional
            Whether to use absolute values.
        normalise: bool, optional
            Whether to normalise.
        normalise_func: callable, optional
            The normalisation function.
        normalise_func_kwargs: dict, optional
            The normalisation function keyword arguments.
        return_aggregate: bool, optional
            Whether to return an aggregate.
        aggregate_func: callable, optional
            The aggregate function.
        default_plot_func: callable, optional
            The default plot function.
        disable_warnings: bool, optional
            Whether to disable warnings.
        display_progressbar: bool, optional
            Whether to display a progress bar.
        debug: bool, optional
            Whether to use debug mode.
        base_seed: int, optional
            The base seed.
        kwargs: optional
            Keyword arguments.
        """
        self.normalise = normalise

        if normalise_func is None:
            normalise_func = normalise_by_average_second_moment_estimate
        self.normalise_func = normalise_func

        if normalise_func_kwargs is None:
            normalise_func_kwargs = {}
        self.normalise_func_kwargs = normalise_func_kwargs

        if pullback_kwargs is None:
            pullback_kwargs = {}
        self.pullback_kwargs = pullback_kwargs

        self.explain_func = explain_gef  # FIXME.

        super().__init__(
            abs=abs,
            normalise=self.normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        self.M = M
        self.Z = Z
        self.perturbation_path = perturbation_path
        self.evaluate_class_distortion = evaluate_class_distortion
        self.input_mode = input_mode
        self.fast_mode = fast_mode
        self.K = K
        self.T = T
        self.activation_noise = activation_noise
        self.num_classes = num_classes
        self.debug = debug
        self.layer_idx = layer_idx
        self.mean = mean
        self.noise_type = noise_type

        # Set the base seed for reproducibility.
        self.base_seed = base_seed
        if self.base_seed is not None:
            random.seed(self.base_seed)
            self.seeds = generate_seeds(num_seeds=10000000)

        if similarity_func is None:
            similarity_func = correlation_spearman
        self.similarity_func = similarity_func

        if model_disortion_func is None:
            model_disortion_func = distance_euclidean
        self.model_disortion_func = model_disortion_func

        if explanation_disortion_func is None:
            explanation_disortion_func = distance_euclidean
        self.explanation_disortion_func = explanation_disortion_func

        # Asserts and warnings.
        if not self.disable_warnings:
            if self.num_classes is None:
                warnings.warn(
                    "Please provide the number of classes ('num_classes') at initialisation. Setting 'num_classes' = 1000."
                )
            self.num_classes = 1000
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "the number of N iterated "
                    "over 'n_f', the function to perturb the input 'perturb_func',"
                    " the similarity metric 'similarity_func' as well as norm "
                    "calculations on the prediction and explanation differenes "
                    "i.e., 'model_disortion_func' and 'explanation_disortion_func'"
                ),
                citation=("INSERT."),
            )

        self.gef_scores = []

        if self.debug:
            self.J_fs = (
                np.zeros(
                    (
                        self.M,
                        self.Z,
                        self.T,
                        self.K,
                        1,
                        1,
                    )
                )
                * np.nan
            )
            self.L_gammas = np.zeros((self.M, self.Z, self.T, 1))
            self.de = (
                np.zeros(
                    (
                        self.M,
                        self.Z,
                        self.T,
                        1,
                        1,
                    )
                )
                * np.nan
            )

    def __call__(
        self,
        model,
        x_batch: Union[np.ndarray, torch.Tensor],
        y_batch: Union[np.ndarray, torch.Tensor],
        a_batch: Optional[Union[np.ndarray, torch.Tensor]] = None,
        s_batch: Optional[Union[np.ndarray, torch.Tensor]] = None,
        channel_first: Optional[bool] = True,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 100,
        custom_batch: Optional[Any] = None,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        last_results: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        self.channel_first = channel_first
        self.softmax = softmax
        self.device = device
        self.batch_size = min(batch_size, x_batch.shape[0])
        self.explain_func = explain_func
        self.model_predict_kwargs = model_predict_kwargs or {}
        self.explain_func_kwargs = explain_func_kwargs or {}
        # self.explain_func_kwargs["softmax"] = softmax

        custom_batch_updated = {}
        custom_batch_updated["am_batch"] = None
        if custom_batch is not None:
            # Passed as numpy to generate the batch (quantus-specific requirement).
            custom_batch_updated["am_batch"] = custom_batch
            if isinstance(custom_batch, np.ndarray):
                custom_batch = torch.tensor(custom_batch, dtype=torch.long).to(device)
            self.model_predict_kwargs["attention_mask"] = custom_batch
            self.explain_func_kwargs["attention_mask"] = custom_batch

        # Generate predicted labels to explain against.
        model = get_wrapped_model_gef(
            model=model,
            channel_first=self.channel_first,
            softmax=self.softmax,
            device=self.device,
            model_predict_kwargs=self.model_predict_kwargs,
        )

        # Make sure explanations are generated against predictions and not ground truth labels.
        y_pred_batch = model.predict(x_batch, **self.model_predict_kwargs)
        custom_batch_updated["y_pred_batch"] = y_pred_batch
        y_pred_class_batch = np.argmax(y_pred_batch, axis=1)

        # Calculate fraction of similar predictions.
        acc_orig = np.mean(
            np.equal(
                (
                    y_batch.detach().cpu().numpy()
                    if isinstance(y_batch, torch.Tensor)
                    else y_batch
                ),
                (
                    y_pred_class_batch.detach().cpu().numpy()
                    if isinstance(y_pred_class_batch, torch.Tensor)
                    else y_pred_class_batch
                ),
            )
        )

        self.accuracy = np.zeros((self.M + 1, self.Z + 1)) * np.nan
        self.acc_random = float(1 / self.num_classes)
        self.accuracy[0, 0] = acc_orig

        # For llm-x explanations.
        if self.explain_func_kwargs["method"].startswith("LLM"):
            self.explain_func_kwargs["logits_original"] = (
                y_pred_batch  # np.max(y_pred_batch, axis=1)
            )
            self.explain_func_kwargs["logits_perturb"] = self.explain_func_kwargs[
                "logits_original"
            ]

        assert (
            y_pred_class_batch.shape == y_batch.shape
        ), f"The shapes of y_pred_batch {y_pred_class_batch.shape} is not the same as y_batch {y_batch.shpae} before batching."

        # Re-compute the perturbation levels if not given (on all samples, not batch_wise).
        if self.perturbation_path is None:
            # TODO. Implement this batch wise.
            self.perturbation_path = self.compute_perturbation_path(
                model=model, x_batch=x_batch, y_batch=y_pred_class_batch
            )
        else:
            self.Z = len(self.perturbation_path)

        # TODO. Only the brige scores are saved fully of all self attributes (rest only).
        return super().__call__(
            model=model,
            x_batch=(
                x_batch.detach().cpu().numpy()
                if isinstance(x_batch, torch.Tensor)
                else x_batch
            ),
            y_batch=y_pred_class_batch,
            a_batch=(
                a_batch.detach().cpu().numpy()
                if isinstance(a_batch, torch.Tensor)
                else a_batch
            ),
            s_batch=s_batch,
            custom_batch=custom_batch_updated,
            channel_first=self.channel_first,
            explain_func=self.explain_func,
            explain_func_kwargs=self.explain_func_kwargs,
            softmax=self.softmax,
            device=self.device,
            model_predict_kwargs=self.model_predict_kwargs,
            batch_size=self.batch_size,
            **kwargs,
        )

    def evaluate_batch(
        self,
        model: ModelInterfaceGEF,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        am_batch: np.ndarray,
        y_pred_batch: np.ndarray,
        custom_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates model and attributes on a single data batch and returns the batched evaluation result.

        Parameters
        ----------
        model: ModelInterfaceGEF
            A ModelInterfaceGEF that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        s_batch: np.ndarray
            The segmentation to be evaluated on a batch-basis.

        Returns
        -------
           np.ndarray
            The batched evaluation results.
        """
        if am_batch is not None:
            self.explain_func_kwargs["attention_mask"] = (
                torch.Tensor(am_batch).to(self.device).long()
            )
            self.model_predict_kwargs["attention_mask"] = (
                torch.Tensor(am_batch).to(self.device).long()
            )

        # Initialise arrays.
        self.attr_shape = np.prod(np.squeeze(a_batch).shape[1:])
        self.distortion_f = np.zeros((self.M, self.Z, self.batch_size)) * np.nan
        self.distortion_e = np.zeros((self.M, self.Z, self.batch_size)) * np.nan

        # Empty cache.
        torch.cuda.empty_cache()

        if self.debug:
            self.J_fs = (
                np.zeros(
                    (
                        self.M,
                        self.Z,
                        self.T,
                        self.K,
                        self.batch_size,
                        self.attr_shape,
                    )
                )
                * np.nan
            )
            self.L_gammas = np.zeros((self.M, self.Z, self.T, self.batch_size))
            self.de = (
                np.zeros(
                    (
                        self.M,
                        self.Z,
                        self.T,
                        self.batch_size,
                        self.attr_shape,
                    )
                )
                * np.nan
            )

        for m_ix in range(self.M):
            length = 0
            prev_std = 1e-4
            for p_ix, std in enumerate(self.perturbation_path):

                self.explain_func_kwargs["magnitude_level"] = get_magnitude_level(
                    p_ix, self.K
                )
                if self.input_mode:

                    # TODO. Implement this.
                    # Check if x_batch is dtype integer or long.
                    # if x_batch.dtype in [np.int, np.int32, np.int64, torch.int, torch.long]:
                    #     # Perturb input with masking.
                    # Convert std to number of indieces of full input size.
                    #     x_batch_perturbed = self.add_masks_to_input(x_batch, std)
                    # else:

                    # Perturb input with Gaussian noise.
                    x_batch_perturbed = self.add_noise_to_input(
                        x_batch, std, mean=self.mean, noise_type=self.noise_type
                    )
                    y_pred_batch_perturbed = model.predict(
                        x_batch_perturbed, **self.model_predict_kwargs
                    )

                    # Generate perturbed explanations.
                    a_batch_p = self.explain_batch(
                        model=model,
                        x_batch=x_batch_perturbed,
                        y_batch=y_batch,
                    )
                    del x_batch_perturbed
                    gc.collect()
                    torch.cuda.empty_cache()

                else:

                    # Perturb model weights, predict and explain.
                    model_p = self.add_noise_to_model(
                        model=model, std=std, mean=self.mean, noise_type=self.noise_type
                    )
                    y_pred_batch_perturbed = model_p.predict(
                        x_batch, **self.model_predict_kwargs
                    )

                    # For llm-x explanations.
                    if self.explain_func_kwargs["method"].startswith("LLM"):
                        self.explain_func_kwargs["logits_perturb"] = (
                            y_pred_batch_perturbed
                        )
                    a_batch_p = self.explain_batch(
                        model=model_p,
                        x_batch=x_batch,
                        y_batch=y_batch,
                    )
                    model_p.get_model().cpu()
                    del model_p
                    gc.collect()
                    torch.cuda.empty_cache()

                # Measure distortions for each sample.
                for s_ix in range(self.batch_size):

                    if self.evaluate_class_distortion:

                        # On a single logit (argmax prediction of unperturbed sample).
                        expl_ix = y_batch[s_ix]
                        distortion_f = self.model_disortion_func(
                            a=y_pred_batch[s_ix, expl_ix].reshape(-1),
                            b=y_pred_batch_perturbed[s_ix, expl_ix].reshape(-1),
                        )
                    else:

                        # On the full output layer i.e., all classes.
                        distortion_f = self.model_disortion_func(
                            a=y_pred_batch[s_ix].reshape(-1),
                            b=y_pred_batch_perturbed[s_ix].reshape(-1),
                        )

                    self.distortion_f[m_ix, p_ix, s_ix] = distortion_f

                    if self.fast_mode:

                        # Measure distortions for explanations.
                        if np.isnan(a_batch_p).all():
                            self.distortion_e[m_ix, p_ix, s_ix] = np.nan

                        else:
                            self.distortion_e[m_ix, p_ix, s_ix] = (
                                self.explanation_disortion_func(
                                    a=a_batch[s_ix].reshape(-1),
                                    b=a_batch_p[s_ix].reshape(-1),
                                )
                            )

                # Measure distortions with pullback, batch-wise.
                if not self.fast_mode:
                    length += self.pullback(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        prev_std=prev_std,
                        std=std,
                        p_ix=p_ix,
                        m_ix=m_ix,
                    )
                    self.distortion_e[m_ix, p_ix, :] = length / ((p_ix + 1) * self.T)

                    gc.collect()
                    torch.cuda.empty_cache()

                y_class_batch_perturbed = np.argmax(y_pred_batch_perturbed, axis=1)

                # Calculate fraction of similar predictions.
                acc_perturbed = np.mean(
                    np.equal(
                        (
                            y_batch.detach().cpu().numpy()
                            if isinstance(y_batch, torch.Tensor)
                            else y_batch
                        ),
                        (
                            y_class_batch_perturbed.detach().cpu().numpy()
                            if isinstance(y_class_batch_perturbed, torch.Tensor)
                            else y_class_batch_perturbed
                        ),
                    )
                )
                self.accuracy[m_ix + 1, p_ix + 1] = acc_perturbed

            gc.collect()
            torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        # Measure similarity between distortions over perturbations for each sample.
        self.scores = np.zeros((self.M, self.batch_size)) * np.nan
        for m_ix in range(self.M):
            for s_ix in range(self.batch_size):
                if not np.isnan(self.distortion_e[m_ix, :, s_ix]).any():
                    sim_score = self.similarity_func(
                        a=self.distortion_f[m_ix, :, s_ix],
                        b=self.distortion_e[m_ix, :, s_ix],
                    )

                self.scores[m_ix, s_ix] = sim_score
        self.gef_scores = np.nanmean(self.scores, axis=0)

        return self.gef_scores

    def custom_preprocess(
        self,
        model: ModelInterfaceGEF,
        x_batch: np.ndarray,
        y_batch: Optional[Union[np.ndarray, torch.Tensor]],
        a_batch: Optional[Union[np.ndarray, torch.Tensor]],
        s_batch: np.ndarray,
        custom_batch: Optional[Union[Dict, np.ndarray, torch.Tensor]],
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        None
        """
        return custom_batch
        # y_pred_batch, attention_mask = custom_batch
        # return {"am_batch": attention_mask, "y_pred_batch": y_pred_batch}

    def compute_perturbation_path(
        self,
        model,
        x_batch: Union[np.ndarray, torch.Tensor],
        y_batch: Union[np.ndarray, torch.Tensor],
        mean: str = 1.0,
        noise_type: str = "multiplicative",
        tolerance_factor=0.1,
        num_runs: int = 5,
        max_iterations: int = 100,
    ):
        """
        Find the maximum noise level that leads to a randomly-behaving model and return a list of noise levels.

        Parameters
        ----------
        self: object
            An instance of the current class.
        model: object
            A trained classification model.
        x_batch: np.ndarray
            A numpy array of input data N of shape `(batch_size, channels, height, width)`.
        y_batch: np.ndarray
            A numpy array of integer labels of shape `(batch_size,)`.

        Returns
        -------
        noise_array : np.ndarray
            A numpy array of floats that are evenly sampled between 0 and the maximum noise level
            that results in a randomly behaving model. The array has a length of `num_N`.
        """
        # Store std_max from each run.
        std_max_values = []
        tolerance = max(tolerance_factor * self.acc_random, 0.01)

        for run in range(num_runs):
            std_max = None
            std = 0.2
            iteration = 0

            while std_max is None and iteration < max_iterations:
                iteration += 1

                # Perturb on input or model.
                if self.input_mode:
                    x_batch_perturbed = self.add_noise_to_input(
                        x_batch, std, mean=mean, noise_type=noise_type
                    )
                    y_pred_batch_perturbed = model.predict(
                        x_batch_perturbed, **self.model_predict_kwargs
                    )  # torch.Tensor(x_batch_perturbed).to(self.device).float()

                else:
                    model_perturbed = self.add_noise_to_model(
                        model=model,
                        std=std,
                        mean=mean,
                        noise_type=noise_type,
                    )
                    y_pred_batch_perturbed = model_perturbed.predict(
                        x_batch, **self.model_predict_kwargs
                    )
                    model_perturbed.get_model().cpu()
                    del model_perturbed

                y_pred_batch_perturbed = np.argmax(y_pred_batch_perturbed, axis=1)

                # Calculate fraction of similar predictions.
                acc_perturbed = np.mean(
                    np.equal(
                        y_batch.astype(int),
                        y_pred_batch_perturbed.astype(int),
                    )
                ).item()

                # If the accuracy is close to random guessing, set std_max and break.
                if abs(acc_perturbed - self.acc_random) <= tolerance:
                    std_max = std
                    break
                else:
                    std += 0.1

            if std_max is None:
                std_max = std
            std_max_values.append(std_max)

        # Average the std_max values across runs.
        std_max_values = [x if x is not None else np.nan for x in std_max_values]
        if all(x is np.nan for x in std_max_values):
            warnings.warn(
                "\n\tWarning: Maximum iterations and runs reached. The model might not be behaving randomly. Setting avg_std_max to 1.0."
            )
            avg_std_max = 1.0
        else:
            avg_std_max = np.nanmean(std_max_values)

        # Generate a list of floats evenly spaced between 1e-4 and avg_std_max.
        noise_array = np.linspace(1e-3, avg_std_max, self.Z)
        std_of_std_max = np.nanstd(std_max_values) if std_max_values else np.nan
        print(
            f"\tModel acc: {acc_perturbed * 100:.2f}% with avg noise level: {avg_std_max:.2f} (±{std_of_std_max:.2f}) "
            f"from values {np.round(np.array(std_max_values), 2)},\n\tcreating noise_array: {np.round(noise_array, 2)}"
        )
        return noise_array

    def add_noise_to_input(
        self, x_batch, std, noise_type="additive", mean=0
    ) -> np.ndarray:
        """
        Add Gaussian noise to a batch of inputs.

        Parameters
        -----------
        x_batch (torch.Tensor or np.ndarray):
            Batch of inputs.
        std (float):
            Standard deviation of the noise.
        noise_type (str):
            Type of noise ('additive' or 'multiplicative').
        mean (float):
            Mean of the Gaussian noise.

        Returns:
            torch.Tensor or np.ndarray: Noisy input.
        """
        if isinstance(x_batch, torch.Tensor):
            distribution = torch.distributions.normal.Normal(loc=mean, scale=std)

            if noise_type == "additive":
                noise = distribution.sample(x_batch.size()).to(x_batch.device)
                noisy_input = x_batch + noise
            elif noise_type == "multiplicative":
                noise = distribution.sample(x_batch.size()).to(x_batch.device)
                noisy_input = x_batch * (1 + noise)

        else:
            noise = np.random.normal(loc=mean, scale=std, size=x_batch.shape)

            if noise_type == "additive":
                noisy_input = x_batch + noise
            elif noise_type == "multiplicative":
                noisy_input = x_batch * (1 + noise)

        return noisy_input

    def add_masks_to_input(self, x_batch, std):
        """
        Add Gaussian noise to a batch of inputs.

        Parameters
        ----------
        x_batch: np.ndarray
            A numpy array of input data N of shape `(batch_size, channels, height, width)`.
        std: float
            The standard deviation of the noise to be applied to the model weights.
        """
        # Assume that x_batch is a torch tensor of longs with input ids representing tokens.
        # pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
        if isinstance(x_batch, torch.Tensor):
            noise = torch.randn_like(x_batch) * std
            noisy_input = x_batch + noise
        else:
            noise = np.random.randn(*x_batch.shape) * std
        return x_batch + noise

    def add_noise_to_model(
        self,
        model,
        std: float,
        mean: float = 1.0,
        noise_type: str = "multiplicative",
        get_model: bool = False,
    ):
        """
        Add Gaussian noise to the model weights.

        Parameters
        ----------
        model: object
            A trained classification model.
        std: float
            The standard deviation of the noise to be applied to the model weights.
        mean: float
            The mean of the Gaussian noise.
        noise_type: str
            The type of noise ('additive' or 'multiplicative').
        get_model: bool
            Indicates whether to return the model or the wrapped model.
        """
        seed = None
        if self.base_seed is not None:
            if not self.seeds:
                raise ValueError(
                    "Ran out of seeds. Increase num_seeds or reduce the number of models."
                )
            seed = self.seeds.pop(0)

        model_perturbed = model.sample(
            mean=mean,
            std=float(std),
            noise_type=noise_type,
            seed=seed,
        )
        if get_model:
            return model_perturbed

        return get_wrapped_model_gef(
            model=model_perturbed,
            channel_first=self.channel_first,
            softmax=self.softmax,
            device=self.device,
            model_predict_kwargs=self.model_predict_kwargs,
        )

    def pullback(
        self,
        model,
        x_batch,
        y_batch,
        a_batch,
        prev_std: float,
        std: float,
        p_ix: int,
        m_ix: int,
    ):
        """
        Perform the pullback and compute the explanation distortion.

        Parameters
        ----------
        model: object
            A trained classification model.
        x_batch: np.ndarray
            A numpy array of input data N of shape `(batch_size, channels, height, width)`.
        a_batch: np.ndarray
            A numpy array of attributions of shape `(batch_size, channels, height, width)`.
        prev_std: float
            The previous noise level (standard deviation).
        std: float
            The standard deviation of the noise to be applied to the model weights.
        p_ix: int
            The index of the noise level.
        m_ix: int
            The index of the model.

        Returns
        -------
        L_gamma: float
            The functional distance between the model and the explanation function.
        """
        L_gamma = np.zeros(self.batch_size)
        J_f = np.zeros((self.T, self.batch_size, self.attr_shape))
        de = np.zeros((self.T, self.batch_size, self.attr_shape))

        ts = np.linspace(prev_std, std, self.T)
        a_batch = np.squeeze(a_batch).reshape(self.batch_size, self.attr_shape)

        for t_ix, t_noise in enumerate(ts):

            model_perturbed = self.add_noise_to_model(
                model=model,
                std=t_noise,
            )

            # For llm-x explanations.
            if self.explain_func_kwargs["method"].startswith("LLM"):
                y_pred_batch_perturbed = model_perturbed.predict(
                    x_batch, **self.model_predict_kwargs
                )
                self.explain_func_kwargs["logits_perturb"] = y_pred_batch_perturbed

            # Generate perturbed explanations.
            a_batch_perturbed = self.explain_batch(
                model=model_perturbed.get_model(), x_batch=x_batch, y_batch=y_batch
            )

            a_batch_perturbed = np.squeeze(a_batch_perturbed).reshape(
                self.batch_size, self.attr_shape
            )

            # Compute the difference in explanation value.
            de[t_ix, :, :] = a_batch - a_batch_perturbed

            Jf_k = np.zeros((self.K, self.batch_size, self.attr_shape))

            for k_ix in range(self.K):

                # Apply tiny additive noise to the neural activities through perturbation.
                model_noisy = ModelAdditiveNoiseActivations(
                    model_perturbed,
                    model_predict_kwargs=self.model_predict_kwargs,
                    noise=self.activation_noise,
                )
                # For llm-x explanations.
                if self.explain_func_kwargs["method"].startswith("LLM"):
                    y_pred_batch_perturbed = model_noisy.predict(
                        x_batch, **self.model_predict_kwargs
                    )
                    self.explain_func_kwargs["logits_perturb"] = y_pred_batch_perturbed

                # Generate perturbed explanations.
                a_batch_perturbed = self.explain_batch(
                    model=model_noisy.get_unwrapped_model(),
                    x_batch=x_batch,
                    y_batch=y_batch,
                )
                a_batch_perturbed = np.squeeze(a_batch_perturbed).reshape(
                    self.batch_size, self.attr_shape
                )

                # Calculate the feature-wise difference in explanation value and accumulate the differences.
                Jf_k[k_ix, :, :] = a_batch - a_batch_perturbed

                del a_batch_perturbed, model_noisy
                gc.collect()
                torch.cuda.empty_cache()

                if self.debug:
                    self.J_fs[m_ix, p_ix, t_ix, k_ix, :, :] = Jf_k[k_ix, :, :]

            if self.debug:
                self.de[m_ix, p_ix, t_ix, :, :] = de[t_ix, :, :]

            # Average the feature-wise explanation differences over N.
            J_f[t_ix, :, :] = np.nanmean(Jf_k, axis=0)

            del Jf_k, model_perturbed
            gc.collect()
            torch.cuda.empty_cache()

        for s_ix in range(self.batch_size):
            l_gamma = 0
            for t_ix, _ in enumerate(ts):

                # Approximate the length of the path.
                B = de[t_ix, s_ix, None, :]  # (1, 784)
                V = (
                    J_f[t_ix, s_ix, None].T @ J_f[t_ix, s_ix, None]
                )  #  (784, 784) g tensor
                l_gamma += ((B @ V) @ B.T).item()
            L_gamma[s_ix] = l_gamma

            if self.debug:
                self.L_gammas[m_ix, p_ix, t_ix, s_ix] = l_gamma / ((p_ix + 1) * self.T)

        return L_gamma


class ModelAdditiveNoiseActivations(torch.nn.Module):
    """
    A wrapper class that adds noise to the activations of a model.
    """

    def __init__(
        self,
        wrapped_model,
        model_predict_kwargs: Dict,
        noise: float = 1e-3,  # had 1e-4
    ):
        """
        Initialise the ModelAdditiveNoiseActivations class.

        Parameters
        ----------
        wrapped_model: torch.nn.Module
            The model to wrap.
        model_predict_kwargs: dict
            Keyword arguments to be passed to the model's predict method.
        noise: float
            The standard deviation of the noise to be added to the activations.
        """
        super(ModelAdditiveNoiseActivations, self).__init__()
        self.wrapped_model = wrapped_model
        self.noise = noise
        self.model_predict_kwargs = model_predict_kwargs
        self.is_additive = True

        # TODO. Remove the need for this.
        if (
            "attention_mask" in self.model_predict_kwargs
            and self.model_predict_kwargs["attention_mask"] is None
        ):
            del self.model_predict_kwargs["attention_mask"]

    def forward(self, x, **kwargs):
        """
        Forward pass through the model with noise added to the activations.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor to the model.
        kwargs: optional
            Keyword arguments.
        """
        # with torch.no_grad():
        # Forward pass through the original model.
        original_output = self.wrapped_model.predict(
            x, **{**self.model_predict_kwargs, **kwargs}
        )

        # Add noise to the output.
        noise = np.random.randn(*original_output.shape) * self.noise
        noisy_output = original_output + noise

        return noisy_output

    def predict(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def get_unwrapped_model(self) -> torch.nn:
        """
        Get the original torch model.
        """
        return self.wrapped_model.get_model()


@staticmethod
def get_magnitude_level(index: int, path_length: int) -> str:
    """
    Get the magnitude level of the perturbation.

    Parameters
    ----------
    index: int
        The index of the perturbation level.
    path_length: int
        The length of the perturbation path.
    """
    # Define thresholds based on the index within the total path length.
    if index < path_length * 1 / 3:
        return "slightly"
    elif index < path_length * 2 / 3:
        return "moderately"
    else:
        return "heavily"


@staticmethod
def generate_seeds(num_seeds: Optional[int] = 10000000) -> List[int]:
    return [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]

"""This module contains an adaptation of the Quantus implementation of the Efficient Model Parameter Randomisation Test metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import sys
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Generator,
    Iterable,
)


import numpy as np
from tqdm.auto import tqdm
from sklearn.utils import gen_batches

from quantus.functions.similarity_func import correlation_spearman, distance_euclidean
from quantus.functions.normalise_func import normalise_by_average_second_moment_estimate
from quantus.helpers import asserts, warn, utils
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)

from .quantus_model_interface import ModelInterfaceEGEF
from .quantus_metric import MetricGEF, get_wrapped_model_gef
from .quantus_explain import explain_gef

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class MetricLayerDistortion(MetricGEF):
    """
    Adaptation of the Efficient MPRT.

    The Efficient MPRT measures replaces the layer-by-layer pairwise comparison
    between e and ˆe of MPRT by instead computing the relative rise in explanation complexity using only
    two model states, i.e., the original- and fully randomised model version


    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Metric Layer Distortion"
    data_applicability = {DataType.IMAGE, DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.RANDOMISATION

    def __init__(
        self,
        model_disortion_func: Optional[Callable] = None,
        explanation_disortion_func: Optional[Callable] = None,
        similarity_func: Optional[Callable] = None,
        layer_order: str = "bottom_up",
        seed: int = 42,
        compute_extra_scores: bool = False,
        skip_layers: bool = False,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        explanation_disortion_func: callable
            A callable that computes the complexity of an explanation.
        explanation_disortion_func_kwargs: dict, optional
            Keyword arguments to be passed to explanation_disortion_func on call.
        similarity_func: callable
            Similarity function applied to compare input and perturbed input, default=correlation_spearman.
        layer_order: string
            Indicated whether the model is randomized cascadingly or independently.
            Set order=top_down for cascading randomization, set order=independent for independent randomization,
            default="independent".
        seed: integer
            Seed used for the random generator, default=42.
        compute_extra_scores: boolean
            Indicates if exta scores should be computed (and stored in a metric attrbute
            (dict) called scores_extra.
        skip_layers: boolean
            Indicates if explanation similarity should be computed only once; between the
            original and fully randomised model, instead of in a layer-by-layer basis.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=True.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_average_second_moment_estimate.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        if explanation_disortion_func is None:
            explanation_disortion_func = distance_euclidean

        if model_disortion_func is None:
            model_disortion_func = distance_euclidean

        if normalise_func is None:
            normalise_func = normalise_by_average_second_moment_estimate

        if normalise_func_kwargs is None:
            normalise_func_kwargs = {}

        if similarity_func is None:
            similarity_func = correlation_spearman

        self.explanation_disortion_func = explanation_disortion_func
        self.model_disortion_func = model_disortion_func
        self.normalise_func = normalise_func
        self.abs = abs
        self.normalise_func_kwargs = normalise_func_kwargs
        self.similarity_func = similarity_func
        self.layer_order = layer_order
        self.seed = seed
        self.compute_extra_scores = compute_extra_scores
        self.skip_layers = skip_layers

        # Results are returned/saved as a dictionary not like in the super-class as a list.
        self.evaluation_scores = {}

        asserts.assert_layer_order(layer_order=self.layer_order)
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "the order of the layer randomisation 'layer_order' (we recommend "
                    "bottom-up randomisation and advice against top-down randomisation) "
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = True,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[Any]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        The content of evaluation_scores will be appended to all_evaluation_scores (list) at the end of
        the evaluation call.

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
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >>> import quantus
            >>> from quantus import LeNet
            >>> import torch

            # Enable GPU.
            >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >>> model = LeNet()
            >>> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >>> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >>> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >>> x_batch, y_batch = iter(test_loader).next()
            >>> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >>> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >>> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >>> metric = Metric(abs=True, normalise=False)
            >>> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """

        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)
        self.batch_size = batch_size
        self.channel_first = channel_first

        if model_predict_kwargs is None:
            model_predict_kwargs = {}
        self.model_predict_kwargs = model_predict_kwargs

        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
        )
        model: ModelInterface = data["model"]  # type: ignore

        # Here _batch refers to full dataset.
        x_full_dataset = data["x_batch"]
        y_full_dataset = data["y_batch"]
        a_full_dataset = data["a_batch"]

        # Reshape input according to model (PyTorch or Keras/Torch).
        x_full_dataset = model.shape_input(
            x=x_full_dataset,
            shape=x_full_dataset.shape,
            channel_first=channel_first,
            batched=True,
        )

        # Results are returned/saved as a dictionary not as a list as in the super-class.
        self.evaluation_scores = {}

        # Get number of iterations from number of layers.
        n_layers = model.random_layer_generator_length
        pbar = tqdm(
            total=n_layers * len(x_full_dataset), disable=not self.display_progressbar
        )
        if self.display_progressbar:
            # Set property to False, so we display only 1 pbar.
            self._display_progressbar = False

        # Get the number of bins for discrete entropy calculation.
        # if "n_bins" not in self.explanation_disortion_func_kwargs:
        #     if a_batch is None:
        #         a_batch = self.explain_batch(
        #             model=model.get_model(),
        #             x_batch=x_full_dataset,
        #             y_batch=y_full_dataset,
        #         )
        #     self.find_n_bins(
        #         a_batch=a_batch,
        #         n_bins_default=self.explanation_disortion_func_kwargs.get("n_bins_default", 100),
        #         min_n_bins=self.explanation_disortion_func_kwargs.get("min_n_bins", 10),
        #         max_n_bins=self.explanation_disortion_func_kwargs.get("max_n_bins", 200),
        #         debug=self.explanation_disortion_func_kwargs.get("debug", False),
        #     )

        self.explanation_distortions_by_layer: Dict[str, List[float]] = {}
        self.model_distortions_by_layer: Dict[str, List[float]] = {}

        with pbar as pbar:
            for l_ix, (layer_name, random_layer_model) in enumerate(
                model.get_random_layer_generator(order=self.layer_order, seed=self.seed)
            ):
                pbar.desc = layer_name

                if l_ix == 0:
                    # Generate explanations on original model in batches.
                    a_original_generator = self.generate_explanations(
                        model.get_model(), x_full_dataset, y_full_dataset, batch_size
                    )

                    # Compute the complexity of explanations of the original model.
                    self.explanation_distortions_by_layer["orig"] = []
                    for a_batch, a_batch_original in zip(
                        self.generate_a_batches(a_full_dataset), a_original_generator
                    ):
                        for a_instance, a_instance_original in zip(
                            a_batch, a_batch_original
                        ):
                            print(np.shape(a_instance), np.shape(a_instance_original))
                            distortion = self.explanation_disortion_func(
                                a=a_instance.reshape(-1),
                                b=a_instance_original.reshape(-1),
                            )
                            self.explanation_distortions_by_layer["orig"].append(
                                distortion
                            )
                            pbar.update(1)

                    # Compute the similarity of outputs of the original model.
                    self.model_distortions_by_layer["orig"] = []
                    y_pred_batch_original = model.predict(
                        x_full_dataset, **self.model_predict_kwargs
                    )
                    y_pred_class_batch_original = np.argmax(
                        y_pred_batch_original, axis=1
                    )

                    for y_ix, y_pred in enumerate(y_pred_batch_original):
                        idx_pred = y_pred_class_batch_original[y_ix]
                        # print(y_pred)
                        y_pred = y_pred[idx_pred]
                        # print(np.shape(y_pred), y_pred, [y_pred])
                        distortion = self.model_disortion_func(a=[y_pred], b=[y_pred])
                        self.model_distortions_by_layer["orig"].append(distortion)

                # Skip layers if computing delta.
                if self.skip_layers and (l_ix + 1) < n_layers:
                    continue

                # Generate explanations on perturbed model in batches.
                a_perturbed_generator = self.generate_explanations(
                    random_layer_model, x_full_dataset, y_full_dataset, batch_size
                )

                # Compute the complexity of explanations of the perturbed model.
                self.explanation_distortions_by_layer[layer_name] = []
                for a_batch_original, a_batch_perturbed in zip(
                    a_original_generator, a_perturbed_generator
                ):
                    for a_instance_original, a_instance_perturbed in zip(
                        a_batch_original, a_batch_perturbed
                    ):
                        print(
                            np.shape(a_instance_original),
                            np.shape(a_instance_perturbed),
                        )
                        distortion = self.explanation_disortion_func(
                            a=a_instance_original.reshape(-1),
                            b=a_instance_perturbed.reshape(-1),
                        )
                        self.explanation_distortions_by_layer[layer_name].append(
                            distortion
                        )
                        pbar.update(1)

                # Wrap the model.
                random_layer_model_wrapped = utils.get_wrapped_model(
                    model=random_layer_model,
                    channel_first=channel_first,
                    softmax=softmax,
                    device=device,
                    model_predict_kwargs=model_predict_kwargs,
                )

                # Predict and save complexity scores of the perturbed model outputs.
                self.model_distortions_by_layer[layer_name] = []
                y_pred_batch = random_layer_model_wrapped.predict(
                    x_full_dataset, **self.model_predict_kwargs
                )

                for y_ix, y_pred_pert in enumerate(y_pred_batch):
                    idx_pred = y_pred_class_batch_original[y_ix]
                    # print("y_pred", y_pred)
                    # print("y_pred_pert", y_pred_pert)
                    y_pred = y_pred_batch_original[y_ix, idx_pred]
                    y_pred_pert = y_pred_pert[idx_pred]
                    # print(
                    #    "y_pred_pert", np.shape(y_pred_pert), y_pred_pert, [y_pred_pert]
                    # )
                    # print("y_pred", np.shape(y_pred), y_pred, [y_pred])
                    distortion = self.model_disortion_func(a=[y_pred], b=[y_pred_pert])
                    self.model_distortions_by_layer[layer_name].append(distortion)

        # Save evaluation scores as the relative rise in complexity.
        explanation_scores = list(self.explanation_distortions_by_layer.values())
        self.evaluation_scores = [None for _ in range(len(x_full_dataset))]

        """
        # Compute extra scores and save the results in metric attributes.
        if self.compute_extra_scores:
            self.scores_extra = {}

            # Compute absolute deltas for explanation scores.
            self.scores_extra["scores_delta_explanation"] = [
                b - a for a, b in zip(explanation_scores[0], explanation_scores[-1])
            ]

            # Compute simple fraction for explanation scores.
            self.scores_extra["scores_fraction_explanation"] = [
                b / a if a != 0 else np.nan
                for a, b in zip(explanation_scores[0], explanation_scores[-1])
            ]

            # Compute absolute deltas for model scores.
            model_scores = list(self.model_distortions_by_layer.values())
            self.scores_extra["scores_delta_model"] = [
                b - a for a, b in zip(model_scores[0], model_scores[-1])
            ]

            # Compute simple fraction for model scores.
            self.scores_extra["scores_fraction_model"] = [
                b / a if a != 0 else np.nan
                for a, b in zip(model_scores[0], model_scores[-1])
            ]

            # Compute delta skill score per sample (model versus explanation).
            self.scores_extra["scores_delta_explanation_vs_models"] = [
                b / a if a != 0 else np.nan
                for a, b in zip(
                    self.scores_extra["scores_fraction_model"],
                    self.scores_extra["scores_fraction_explanation"],
                )
            ]
            # Compute the average complexity scores, per sample.
            self.scores_extra["scores_average_complexity"] = (
                self.recompute_average_complexity_per_sample()
            )

            # Compute the correlation coefficient between the model and explanation complexity, per sample.
            self.scores_extra["scores_correlation_model_vs_explanation_complexity"] = (
                self.recompute_model_explanation_correlation_per_sample()
            )
        """

        if self.return_aggregate:
            self.evaluation_scores = [self.aggregate_func(self.evaluation_scores)]

        # Return all_evaluation_scores according to Quantus.
        self.all_evaluation_scores.append(self.evaluation_scores)

        return self.evaluation_scores

    def evaluate_instance(
        self,
        model: MetricGEF,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        s: Optional[np.ndarray],
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
        """
        # Compute complexity measure.
        return None
        # return self.explanation_disortion_func(a=a, b=b)  # **self.explanation_disortion_func_kwargs

    def custom_preprocess(
        self,
        model: MetricGEF,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray],
        **kwargs,
    ) -> Optional[Dict[str, np.ndarray]]:
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
        kwargs:
            Unused.
        Returns
        -------
        None
        """
        # Additional explain_func assert, as the one in general_preprocess()
        # won't be executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)
        if a_batch is not None:  # Just to silence mypy warnings
            return None

        a_batch_chunks = []
        for a_chunk in self.generate_explanations(
            model, x_batch, y_batch, self.batch_size
        ):
            a_batch_chunks.extend(a_chunk)
        return dict(a_batch=np.asarray(a_batch_chunks))

    def generate_explanations(
        self,
        model: MetricGEF,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        batch_size: int,
    ) -> Generator[np.ndarray, None, None]:
        """
        Iterate over dataset in batches and generate explanations for complete dataset.

        Parameters
        ----------
        model: ModelInterface
            A ModelInterface that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        kwargs: optional, dict
            List of hyperparameters.

        Returns
        -------
        a_batch:
            Batch of explanations ready to be evaluated.
        """
        for i in gen_batches(len(x_batch), batch_size):
            x = x_batch[i.start : i.stop]
            y = y_batch[i.start : i.stop]
            a = self.explain_batch(model, x, y)
            yield a

    def generate_a_batches(self, a_full_dataset):
        for batch in gen_batches(len(a_full_dataset), self.batch_size):
            yield a_full_dataset[batch.start : batch.stop]

    def evaluate_batch(self, *args, **kwargs):
        raise RuntimeError(
            "`evaluate_batch` must never be called for `Model Parameter Randomisation Test`."
        )

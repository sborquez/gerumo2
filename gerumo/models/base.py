from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

import sklearn
import numpy as np
import tensorflow as tf
from fvcore.common.registry import Registry

from gerumo.data.constants import TELESCOPES
from ..config.config import configurable, get_cfg
from ..utils.structures import (
    Event, InputShape, Observations, Pointing, ReconstructionMode, Task, Telescope
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for Models.
The registered object will be called with `obj(cfg)`.
The call is expected to return an :class:`BaseModel`.
"""


def build_model(cfg, input_shape) -> Union['BaseModel', 'SKLearnModel']:
    """
    Build Models defined by `cfg.MODEL.ARCHITECTURE.NAME`.
    """
    name = cfg.MODEL.ARCHITECTURE.NAME
    return MODEL_REGISTRY.get(name)(cfg, input_shape)


class LoadableModel:

    _KWARGS = []

    @configurable
    def __init__(self, input_shape: InputShape, mode: ReconstructionMode,
                 task: Task, telescopes: List[Telescope], pointing: Union[tuple, Pointing],
                 weights: Optional[str] = None, **kwargs):
        super(LoadableModel, self).__init__()
        assert (mode is ReconstructionMode.SINGLE and len(telescopes) == 1) \
            or (mode is ReconstructionMode.STEREO)
        self.mode = mode
        self.task = task
        self.pointing = pointing
        self.telescopes = telescopes
        self.weights_path = weights
        self._input_shape = input_shape
        self.enable_fit_mode = False
        self._model = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        config = {
            'input_shape': input_shape,
            'mode': ReconstructionMode[cfg.MODEL.RECONSTRUCTION_MODE],
            'task': Task[cfg.MODEL.TASK],
            'telescopes': [TELESCOPES[tel] for tel in cfg.MODEL.TELESCOPES],
            'pointing': Pointing(*cfg.DATASETS.POINTING),
            'weights': cfg.MODEL.WEIGHTS
        }
        config.update({
            k: v for k, v in cfg.MODEL.ARCHITECTURE.KWARGS if k in cls._KWARGS
        })
        return config

    def _get_model(self):
        if self._model is None:
            x = [tf.keras.Input(shape=s_i[1:]) for s_i in self._input_shape.get()]
            x = x[0] if len(x) == 1 else x
            # TODO: check list [x] vs x
            try:
                self._model = tf.keras.Model(inputs=[x], outputs=self.call(x))
            except:
                self._model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return self._model

    def summary(self):
        return self._get_model().summary()
    
    def plot(self, *args, **kwargs):
        return tf.keras.utils.plot_model(
            self._get_model(),
            *args, **kwargs
        )

    def fit_mode(self):
        self.enable_fit_mode = True

    def evaluation_mode(self):
        self.enable_fit_mode = False

    def preprocess_input(self, inputs: List[Observations]):
        """Convert list of observations into array input"""
        if isinstance(inputs[0], Observations):
            return Observations.list_to_tensor(self.mode, inputs)
        return inputs

    def preprocess_output(self, outputs: List[Event]):
        """Convert list of events into array output"""
        if isinstance(outputs[0], Event):
            return Event.list_to_tensor(outputs)
        return outputs

    def point_estimation(self, predictions):
        """Convert the models regression output into a point form.
        
        This method is handy for custom output formats like umonne or bmo
        """
        return predictions

    def categorical_estimation(self, predictions):
        """Convert the models regression output into a point form.
        
        This method is handy for custom output formats like umonne or bmo
        """
        return tf.reshape(tf.math.argmax(predictions, axis=-1), (-1, 1))

    def postprocess_output(self, predictions):
        """Convert output tensor into a prediction."""
        if self.task is Task.REGRESSION:
            # Convert into a vector
            return self.point_estimation(predictions)
        elif self.task is Task.CLASSIFICATION:
            # Convert into categorical
            return self.categorical_estimation(predictions)
        else:
            raise NotImplementedError


class BaseModel(LoadableModel, tf.keras.Model):

    _KWARGS = []
    REGRESSION_OUTPUT_TYPE = None
    CLASSIFICATION_OUTPUT_TYPE = None
    
    @configurable
    def __init__(self, input_shape: InputShape, mode: ReconstructionMode,
                 task: Task, telescopes: List[Telescope], pointing: Union[tuple, Pointing],
                 weights: Optional[str] = None, **kwargs):
        LoadableModel.__init__(
            self, input_shape, mode, task, telescopes, pointing, weights, **kwargs
        )
        tf.keras.Model.__init__(self)
        self.architecture(**kwargs)

    def call(self, inputs: Union[List[Observations], Any], training: bool = False, uncertainty: bool = False):
        if training or self.enable_fit_mode:
            X = inputs
            y = self.forward(X, training)
            if uncertainty:
                logger.warn('Uncertainty can be computed during training.')
            return y
        else:
            X = self.preprocess_input(inputs)
            y = self.forward(X, training)
            prediction = self.postprocess_output(y)
            if uncertainty:
                uncertainty = self.uncertainty(y)
                return prediction, y, uncertainty
            return prediction
    
    def uncertainty(self, y_pred: tf.Tensor) -> tf.Tensor:
        if self.task is Task.REGRESSION:
            # Compute Variance
            return self.compute_variance(y_pred)
        elif self.task is Task.CLASSIFICATION:
            # Compute predictive entropy
            return self.compute_predictive_entropy(y_pred)
        else:
            raise NotImplementedError
    
    @abstractmethod
    def compute_variance(self, y_pred: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def compute_predictive_entropy(self, y_pred: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def architecture(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, X, training=False):
        pass

    @abstractmethod
    def get_output_dim(self):
        pass


class SKLearnModel(LoadableModel, sklearn.base.BaseEstimator):

    _KWARGS = []

    @configurable
    def __init__(self, input_shape: InputShape, mode: ReconstructionMode,
                 task: Task, telescopes: List[Telescope],
                 weights: Optional[str] = None, **kwargs):
        LoadableModel.__init__(
            self, input_shape, mode, task, telescopes, weights, **kwargs
        )
        sklearn.base.BaseEstimator.__init__(self)
        self.estimator = None
        self.encoder = None
        self.get_estimator(weights, **kwargs)
        assert self.estimator is not None, 'estimator is `None`'

    def preprocess_output(self, outputs):
        outputs = super().preprocess_output(outputs)
        if self.task is Task.CLASSIFICATION:
            # one-hot encoding to categorical
            outputs = self.encoder.fit_transform(outputs).toarray()
        return outputs

    def postprocess_output(self, outputs):
        if self.task is Task.CLASSIFICATION:
            # one-hot encoding to categorical
            outputs = outputs.argmax(axis=-1).reshape((-1, 1))
        # return super().postprocess_output(outputs)
        return outputs

    def fit(self, inputs, outputs):
        X = self.preprocess_input(inputs)
        y = self.preprocess_output(outputs)
        self.estimator.fit(X, y)

    def __call__(self, inputs: Union[List[Observations], Any]):
        X = self.preprocess_input(inputs)
        y = self.estimator.predict(X)
        return self.postprocess_output(y)

    @abstractmethod
    def get_estimator(self, weights, **kwargs):
        pass


ENSEMBLER_REGISTRY = Registry("ENSEMBLER")
ENSEMBLER_REGISTRY.__doc__ = """
Registry for Ensemblers.
The registered object will be called with `obj(cfg)`.
The call is expected to return an :class:`BaseEnsembler`.
"""


def build_ensembler(cfg, input_shapes) -> 'BaseEnsembler':
    """
    Build Ensemblers defined by `cfg.ENSEMBLER.NAME`.
    """
    name = cfg.ENSEMBLER.NAME
    return ENSEMBLER_REGISTRY.get(name)(cfg, input_shapes)


class BaseEnsembler:
    
    @configurable
    def __init__(self, input_shapes: Dict[Telescope, InputShape], task: Task,
                 models: Dict[Telescope, BaseModel], telescopes: List[Telescope],
                 pointing: Union[tuple, Pointing], weights: Optional[Dict[Telescope, str]] = None, **kwargs):
        self.mode = ReconstructionMode.STEREO
        self.task = task
        self.pointing = pointing
        self.telescopes = telescopes
        self.weights_paths = weights
        self.input_shapes = input_shapes
        self.enable_fit_mode = False
        self.models = models

    @classmethod
    def from_config(cls, cfg, input_shapes):
        config = {
            'input_shapes': input_shapes,
            'task': Task[cfg.MODEL.TASK],
            'pointing': Pointing(*cfg.DATASETS.POINTING),
        }
        config['telescopes'] = []
        config['models'] = {}
        config['weights'] = {}
        for telelescope_type, model_cfg_file, model_weight in zip(cfg.ENSEMBLER.TELESCOPES, cfg.ENSEMBLER.ARCHITECTURES, cfg.ENSEMBLER.WEIGHTS):
            telescope = TELESCOPES[telelescope_type]
            config['telescopes'].append(telescope)
            config['models'][telescope] = cls._load_model(model_cfg_file, model_weight, telescope, input_shapes[telescope])
            config['weights'][telescope] = model_weight
        return config

    @classmethod
    def _load_model(cls, model_cfg_file, model_weights, telescope, input_shape):
        model_cfg = get_cfg()
        model_cfg.merge_from_file(model_cfg_file)
        model_cfg.freeze()
        assert telescope.type == model_cfg.MODEL.TELESCOPES[0]
        model = build_model(model_cfg, input_shape)
        model.fit_mode()
        model(cls._model_sample(input_shape))
        if model_weights is not None:
            try:
                model.load_weights(model_weights)
            except ValueError:
                model._get_model()
                model.load_weights(model_weights)
        model.evaluation_mode()
        return model
    
    @classmethod
    def _model_sample(cls, input_shape):
        X = []
        if input_shape.has_image():
            X.append(
                np.random.random((1, *input_shape.images_shape[1:]))
            )
        if input_shape.has_features():
            X.append(
                np.random.random((1, *input_shape.features_shape[1:]))
            )
        return X[0] if len(X) == 1 else tuple(X)
    
    @abstractmethod
    def ensemble(self, models_outputs: tf.Tensor) -> tf.Tensor:
        """Combine the outputs for a single event"""
        pass

    def postprocess_output(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Convert output tensor into a prediction."""
        if self.task is Task.REGRESSION:
            # Convert into a vector
            return self.point_estimation(y_ensembled)
        elif self.task is Task.CLASSIFICATION:
            # Convert into categorical
            return self.categorical_estimation(y_ensembled)
        else:
            raise NotImplementedError

    @abstractmethod
    def point_estimation(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Convert the ensembler regression output into a point form.
        
        This method is handy for custom output formats like umonne or bmo
        """
        raise NotImplementedError

    @abstractmethod
    def categorical_estimation(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Convert the ensembler regression output into a point form.
        
        This method is handy for custom output formats like umonne or bmo
        """
        raise NotImplementedError

    def uncertainty(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        if self.task is Task.REGRESSION:
            # Compute Variance
            return self.compute_variance(y_ensembled)
        elif self.task is Task.CLASSIFICATION:
            # Compute predictive entropy
            return self.compute_predictive_entropy(y_ensembled)
        else:
            raise NotImplementedError
    
    @abstractmethod
    def compute_variance(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Variance for regression output."""
        raise NotImplementedError

    @abstractmethod
    def compute_predictive_entropy(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Predictive entropy for classification output."""
        raise NotImplementedError

    def __call__(self, X: List[Observations], uncertainty: bool = True) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """Compute the predictions for each observation and ensemble the predictions
        into a single output.

        Args:
            X (List[Observations]): Stereo observations with multiples images
                per event.
            uncertainty (bool, optional): If is True, compute the uncertainty and
                return the raw prediction `y_ensembled`, the postprocess output
                `ensembled_prediction` and the `ensembled_uncertainty`.
                Defaults to `True`.

        Returns:
            Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                If `uncertainty` is `True`:
                    `ensembled_prediction`, `y_ensembled`, `ensembled_uncertainty`
                else:
                    `ensembled_prediction`
        """
        X_by_telescope = Observations.list_to_tensor(self.mode, X, self.telescopes)
        # For each event
        ensembled_predictions = []
        ys_ensembled = []
        ensembled_uncertainties = []
        for X_by_telescope_event_i in X_by_telescope:
            # Telescope model predictions
            models_outputs = []
            for telescope in self.telescopes:
                # Get the observations of a single telescope type
                telescope_inputs = X_by_telescope_event_i[telescope]
                if telescope_inputs is None:
                    continue
                # Predict the raw output with the corresponding model
                _, y, _ = self.models[telescope](telescope_inputs, uncertainty=True)
                models_outputs.append(y)
            # Ensemble all predictions into a single prediction
            models_outputs = tf.concat(models_outputs, axis=0)
            y_ensembled = self.ensemble(models_outputs)
            # Compute point estimation
            ensembled_prediction = self.postprocess_output(y_ensembled)
            # Return raw prediction and uncertainty?
            if uncertainty:
                # Compute uncertainty of the ensembled prediction
                ensembled_uncertainty = self.uncertainty(y_ensembled)
                # Store event prediction
                ensembled_predictions.append(ensembled_prediction)
                ys_ensembled.append(y_ensembled)
                ensembled_uncertainties.append(ensembled_uncertainty)
            else:
                # Store event prediction
                ensembled_predictions.append(ensembled_prediction)
        if uncertainty:
            return tf.concat(ensembled_predictions, axis=0), tf.concat(ys_ensembled, axis=0), tf.concat(ensembled_uncertainties, axis=0)
        return tf.concat(ensembled_predictions, axis=0)

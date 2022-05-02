from abc import abstractmethod
from typing import Any, List, Optional, Union
import logging
from pathlib import Path

import tensorflow as tf
import sklearn
from fvcore.common.registry import Registry

from gerumo.data.generators import BaseGenerator
from gerumo.data.constants import TELESCOPES
from ..config.config import configurable
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
            self._model = tf.keras.Model(inputs=[x], outputs=self.call(x))
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
    
    def uncertainty(self, y_pred: tf.Tensor):
        if self.task is Task.REGRESSION:
            # Compute Variance
            return self.compute_variance(y_pred)
        elif self.task is Task.CLASSIFICATION:
            # Compute predictive entropy
            return self.compute_predictive_entropy(y_pred)
        else:
            raise NotImplementedError
    
    @abstractmethod
    def compute_variance(self, y_pred: tf.Tensor):
        pass

    @abstractmethod
    def compute_predictive_entropy(self, y_pred: tf.Tensor):
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


def build_ensembler(cfg) -> 'BaseEnsembler':
    """
    Build Ensemblers defined by `cfg.ENSEMBLER.NAME`.
    """
    name = cfg.ENSEMBLER.NAME
    return ENSEMBLER_REGISTRY.get(name)(cfg)


class BaseEnsembler:
    pass


def load_model(model: BaseModel, generator: BaseGenerator, output_dir: Union[Path, str], epoch_idx: int = -1):
    """Load model's weights from a checkpoint at given epoch."""
    checkpoints = sorted(list(Path(output_dir).glob('weights/*.h5')))
    if len(checkpoints) == 0:
        raise ValueError('Model doesn`t have checkpoints')
    if epoch_idx >= 0 and len(checkpoints) <= epoch_idx:
        raise ValueError(f'Epoch_idx should be in [0, {len(checkpoints)}) or -1: {epoch_idx}')
    checkpoint = checkpoints[epoch_idx]
    X = generator[0][0]
    _ = model(X)
    try:
        model.load_weights(checkpoint)
    except ValueError:
        model._get_model()
        model.load_weights(checkpoint)
    return model
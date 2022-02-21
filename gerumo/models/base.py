from abc import abstractmethod
from typing import List, Optional, Union, Any
import logging

import tensorflow as tf
import sklearn
from fvcore.common.registry import Registry

from gerumo.data.constants import TELESCOPES
from ..config.config import configurable
from ..utils.structures import (
    Event, InputShape, Observations, ReconstructionMode, Task, Telescope
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
                 task: Task, telescopes: List[Telescope],
                 weights: Optional[str] = None, **kwargs):
        super(LoadableModel, self).__init__()
        assert (mode is ReconstructionMode.SINGLE and len(telescopes) == 1) \
            or (mode is ReconstructionMode.STEREO)
        self.mode = mode
        self.task = task
        self.telescopes = telescopes
        self.weights_path = weights
        self._input_shape = input_shape
        self.enable_fit_mode = False
        self.__model = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        config = {
            'input_shape': input_shape,
            'mode': ReconstructionMode[cfg.MODEL.RECONSTRUCTION_MODE],
            'task': Task[cfg.MODEL.TASK],
            'telescopes': [TELESCOPES[tel] for tel in cfg.MODEL.TELESCOPES],
            'weights': cfg.MODEL.WEIGHTS
        }
        config.update({
            k: v for k, v in cfg.MODEL.ARCHITECTURE.KWARGS if k in cls._KWARGS
        })
        return config

    def _get_model(self):
        if self.__model is None:
            x = [tf.keras.Input(shape=s_i[1:]) for s_i in self._input_shape.get()]
            self.__model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return self.__model

    def summary(self):
        return self._get_model().summary()
    
    def plot(self, *args, **kwargs):
        return tf.keras.utils.plot_model(
            self._get_model(),
            *args, **kwargs
        )

    def fit_mode(self):
        self.enable_fit_mode = True

    def verbose_mode(self):
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

    def postprocess_output(self, outputs, inputs: List[Event]) -> List[Event]:
        """Convert keras model's output into list of Events."""
        # TODO: Convert into a list of Events
        return outputs


class BaseModel(LoadableModel, tf.keras.Model):

    _KWARGS = []

    @configurable
    def __init__(self, input_shape: InputShape, mode: ReconstructionMode,
                 task: Task, telescopes: List[Telescope],
                 weights: Optional[str] = None, **kwargs):
        LoadableModel.__init__(
            self, input_shape, mode, task, telescopes, weights, **kwargs
        )
        tf.keras.Model.__init__(self)
        self.architecture(**kwargs)

    def call(self, inputs: Union[List[Observations], Any], training: bool = False):
        if training or self.enable_fit_mode:
            X = inputs
            y = self.forward(X, training)
            return y
        else:
            X = self.preprocess_input(inputs)
            y = self.forward(X, training)
            return self.postprocess_output(y, X)

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

    def postprocess_output(self, outputs) -> List[Event]:
        if self.task is Task.CLASSIFICATION:
            # one-hot encoding to categorical
            outputs = outputs.argmax(axis=-1).reshape((-1, 1))
        return super().postprocess_output(outputs)

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

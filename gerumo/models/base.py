from abc import abstractmethod
from typing import List, Optional, Union, Any
import logging
import tensorflow as tf
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


def build_model(cfg, input_shape) -> 'BaseModel':
    """
    Build Models defined by `cfg.MODEL.ARCHITECTURE.NAME`.
    """
    name = cfg.MODEL.ARCHITECTURE.NAME
    return MODEL_REGISTRY.get(name)(cfg, input_shape)


class BaseModel(tf.keras.Model):

    _KWARGS = []

    @configurable
    def __init__(self, input_shape: InputShape, mode: ReconstructionMode,
                 task: Task, telescopes: List[Telescope],
                 weights: Optional[str] = None, **kwargs):
        super(BaseModel, self).__init__()
        assert (mode is ReconstructionMode.SINGLE and len(telescopes) == 1) \
            or (mode is ReconstructionMode.STEREO)
        self.mode = mode
        self.task = task
        self.telescopes = telescopes
        self.weights_path = weights
        self._input_shape = input_shape
        self.enable_fit_mode = False
        self.architecture(**kwargs)

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
                k: v for k, v in cfg.MODEL.ARCHITECTURE.KWARGS
                if k in cls._KWARGS
        })
        return config

    def fit_mode(self):
        self.enable_fit_mode = True

    def verbose_mode(self):
        self.enable_fit_mode = False

    def preprocess_input(self, inputs: List[Observations]):
        """Convert list of observations into keras model's input"""
        X = Observations.list_to_tensor(self.mode, inputs)
        return X

    def postprocess_output(self, outputs) -> List[Event]:
        """Convert keras model's output into list of Events."""
        # TODO: Convert into a list of Events
        return outputs

    def call(self, inputs: Union[List[Observations], Any], training: bool = False):
        if training or self.enable_fit_mode:
            X = inputs
            y = self.forward(X, training)
            return y
        else:
            X = self.preprocess_input(inputs)
            y = self.forward(X, training)
            return self.postprocess_output(y)

    @abstractmethod
    def architecture(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, X, training=False):
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

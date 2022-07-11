"""
Data generator to feed models
=============================

This modele define generator (keras.Sequential) to feed differents
models, with their defined input format.
"""

import logging
from abc import abstractmethod
from typing import Any, List, Tuple, Union, Optional

import tensorflow as tf
import numpy as np
import pandas as pd
from fvcore.common.registry import Registry

from gerumo.data.constants import TELESCOPES
from ..config.config import configurable
from ..utils.structures import (
    Event, InputShape, Observations, ReconstructionMode, Telescope
)
from ..data.mappers import (
    InputMapper, OutputMapper, build_input_mapper, build_output_mapper
)
# from sklearn import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


GENERATOR_REGISTRY = Registry("GENERATOR")
GENERATOR_REGISTRY.__doc__ = """
Registry for Generators.
The registered object will be called with `obj(cfg, dataset)`.
The call is expected to return an :class:`BaseGenerator`.
"""


def build_generator(cfg, dataset, subset: Optional[str] = None) -> Union['BaseGenerator', Tuple[tf.data.Dataset, Any, int]]:
    """
    Build Generators defined by `cfg.GENERATOR.NAME`.
    """
    name = cfg.GENERATOR.NAME
    if subset is not None:
        return GENERATOR_REGISTRY.get(name)(cfg, dataset, subset)
    return GENERATOR_REGISTRY.get(name)(cfg, dataset)


class BaseGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 dataset: pd.DataFrame,
                 batch_size: int,
                 input_mapper: InputMapper,
                 output_mapper: OutputMapper,
                 shuffle: bool) -> None:
        super().__init__()
        # Generator parameters
        self.dataset = dataset
        self.size = len(self.dataset)
        self.batch_size = batch_size
        self.length = int(np.floor(len(self.dataset) / self.batch_size))
        self.indexes = np.arange(self.size)
        self.shuffle = shuffle
        # How to generate input
        self.input_mapper = input_mapper
        # How to generate output
        self.output_mapper = output_mapper
        # Enable keras generator mode
        self.enable_fit_mode = False

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return self.length

    def fit_mode(self):
        self.enable_fit_mode = True

    def verbose_mode(self):
        self.enable_fit_mode = False

    def __getitem__(self,
                    index: int
                    ) -> Tuple[List[Observations], List[Event]]:
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        observations, events = self._data_generation(indexes)
        if self.enable_fit_mode:
            return self.to_tensors(observations, events)
        return observations, events

    @abstractmethod
    def to_tensors(self, observations, events):
        raise NotImplementedError

    @abstractmethod
    def get_input_shape(self) -> Union[InputShape, Any]:
        raise NotImplementedError

    @abstractmethod
    def from_config(cls, cfg, dataset):
        raise NotImplementedError

    @abstractmethod
    def _data_generation(self,
                         list_indexes: np.ndarray
                         ) -> Tuple[List[Observations], List[Event]]:
        raise NotImplementedError


@GENERATOR_REGISTRY.register()
class MonoGenerator(BaseGenerator):

    @configurable
    def __init__(self,
                 dataset: pd.DataFrame,
                 telescope: Telescope,
                 batch_size: int,
                 input_mapper: InputMapper,
                 output_mapper: OutputMapper,
                 shuffle: bool = True,
                 strict_shuffle: bool = False) -> None:
        mask = np.logical_and.reduce(
            dataset[["name", "type", "camera_type"]] == telescope.description,
            axis=1)
        dataset = dataset[mask]
        super().__init__(
            dataset, batch_size, input_mapper, output_mapper, shuffle
        )
        self.strict_shuffle = strict_shuffle
        self.telescope = telescope
        self.on_epoch_end()

    @classmethod
    def from_config(cls, cfg, dataset):
        return {
            "dataset": dataset,
            "telescope": TELESCOPES[cfg.MODEL.TELESCOPES[0]],
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "input_mapper": build_input_mapper(cfg),
            "output_mapper": build_output_mapper(cfg),
            "shuffle": cfg.GENERATOR.ENABLE_SHUFFLE,
            "strict_shuffle": cfg.GENERATOR.USE_STRICT_SHUFFLE,
        }

    def to_tensors(self, observations, events):
        observations = Observations.list_to_tensor(
            ReconstructionMode.SINGLE, observations
        )
        events = Event.list_to_tensor(events)
        return observations, events

    def get_input_shape(self) -> Union[InputShape, Any]:
        prev_mode = self.enable_fit_mode
        self.verbose_mode()
        obs_tensors = self[0][0][0].to_tensor()
        images_shape, features_shape = None, None
        # it has image
        if self.input_mapper.image_channels:
            # image is always the first input
            images_shape = (None, *(obs_tensors[0].shape))
        # it has features
        if self.input_mapper.telescope_features:
            # image is always the last input
            features_shape = (None, *(obs_tensors[-1].shape))
        input_shape = InputShape(images_shape, features_shape)
        input_shape.set_batch_size(self.batch_size)
        self.enable_fit_mode = prev_mode
        return input_shape

    def on_epoch_end(self) -> np.ndarray:
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.size)
        if self.shuffle:
            if self.strict_shuffle:
                raise NotImplementedError
            else:
                np.random.shuffle(self.indexes)
        return self.indexes

    def _data_generation(self,
                         list_indexes: np.ndarray
                         ) -> Tuple[List[Observations], List[Event]]:
        'Generates data containing batch_size samples'
        batch_observations = []
        batch_events = []
        for obs_idx in list_indexes:
            obs_df = self.dataset.iloc[[obs_idx]]
            observations = self.input_mapper(obs_df)
            event = self.output_mapper(obs_df)
            batch_events.append(event)
            batch_observations.append(observations)
        return (batch_observations, batch_events)


@GENERATOR_REGISTRY.register()
class OneBatchMonoGenerator(MonoGenerator):

    @configurable
    def __init__(self,
                 dataset: pd.DataFrame,
                 telescope: Telescope,
                 input_mapper: InputMapper,
                 output_mapper: OutputMapper,
                 shuffle: bool = True,
                 strict_shuffle: bool = False) -> None:
        mask = np.logical_and.reduce(
            dataset[["name", "type", "camera_type"]] == telescope.description,
            axis=1)
        dataset = dataset[mask]
        batch_size = len(dataset)
        super().__init__(
            dataset, telescope, batch_size, input_mapper, output_mapper,
            shuffle, strict_shuffle
        )

    @classmethod
    def from_config(cls, cfg, dataset):
        return {
            "dataset": dataset,
            "telescope": TELESCOPES[cfg.MODEL.TELESCOPES[0]],
            "input_mapper": build_input_mapper(cfg),
            "output_mapper": build_output_mapper(cfg),
            "shuffle": cfg.GENERATOR.ENABLE_SHUFFLE,
            "strict_shuffle": cfg.GENERATOR.USE_STRICT_SHUFFLE,
        }

    def get_batch(self):
        return self[0]


@GENERATOR_REGISTRY.register()
class MultiStereoGenerator(BaseGenerator):

    @configurable
    def __init__(self,
                 dataset: pd.DataFrame,
                 telescopes: List[Telescope],
                 # batch_size: int,
                 input_mapper: InputMapper,
                 output_mapper: OutputMapper,
                 shuffle: bool = True,
                 strict_shuffle: bool = False) -> None:
        mask = np.zeros(len(dataset), dtype=bool)
        for telescope in telescopes:
            mask = np.logical_or(
                mask,
                np.logical_and.reduce(
                    dataset[['name', 'type', 'camera_type']] == telescope.description,
                    axis=1
                )
            )
        dataset = dataset[mask]
        super().__init__(
            dataset, 1, input_mapper, output_mapper, shuffle
        )
        self.dataset_by_events = dataset.groupby('event_unique_id')
        self.events = list(self.dataset_by_events.groups)
        self.size = len(self.dataset_by_events)
        self.length = len(self.dataset_by_events)
        self.indexes = np.arange(self.size)
        self.strict_shuffle = strict_shuffle
        self.telescopes = telescopes
        self.on_epoch_end()

    @classmethod
    def from_config(cls, cfg, dataset):
        return {
            'dataset': dataset, 
            'telescopes': [TELESCOPES[telescope] for telescope in cfg.ENSEMBLER.TELESCOPES],
            'input_mapper': build_input_mapper(cfg, ensemble=True),
            'output_mapper': build_output_mapper(cfg),
            'shuffle': cfg.GENERATOR.ENABLE_SHUFFLE,
            'strict_shuffle': cfg.GENERATOR.USE_STRICT_SHUFFLE,
        }

    def to_tensors(self, observations, events):
        observations = Observations.list_to_tensor(
            ReconstructionMode.STEREO, observations
        )
        events = Event.list_to_tensor(events)
        return observations, events

    def get_input_shape(self) -> Union[InputShape, Any]:
        input_shapes = {}
        _defaults = {
            'SST': (48, 48),
            'MST': (84, 28),
            'LST': (55, 47)
        }
        for telescope in self.telescopes:
            telescope_type = telescope.type
            images_shape, features_shape = None, None
            # it has image
            if self.input_mapper.image_channels:
                # image is always the first input
                n_channels = len(self.input_mapper.image_channels)
                image_size = _defaults[telescope_type]
                images_shape = (-1, *image_size, n_channels)
            # it has features
            if self.input_mapper.telescope_features:
                # image is always the last input
                n_features = len(self.input_mapper.telescope_features)
                features_shape = (-1, n_features)
            input_shape = InputShape(images_shape, features_shape)
            input_shapes[telescope] = input_shape
        return input_shapes

    def on_epoch_end(self) -> np.ndarray:
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.size)
        if self.shuffle:
            if self.strict_shuffle:
                raise NotImplementedError
            else:
                np.random.shuffle(self.indexes)
        return self.indexes

    def _data_generation(self,
                         list_indexes: np.ndarray
                         ) -> Tuple[List[Observations], List[Event]]:
        'Generates data containing batch_size samples'
        event_idx = list_indexes[0]
        event = self.events[event_idx]
        event_df = self.dataset_by_events.get_group(event)
        
        batch_observations = [self.input_mapper(event_df)]
        batch_events = [self.output_mapper(event_df)]
 
        return (batch_observations, batch_events)


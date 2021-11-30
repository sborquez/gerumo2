"""
Data generator to feed models
=============================

This modele define generator (keras.Sequential) to feed differents
models, with their defined input format.
"""

import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from fvcore.common.registry import Registry
from typing import List, Tuple

from gerumo.data.constants import TELESCOPES
from ..config.config import configurable
from ..utils.structures import Event, Observations, Telescope
from ..data.mappers import (
    InputMapper, OutputMapper, build_input_mapper, build_output_mapper
)
# from sklearn import utils

GENERATOR_REGISTRY = Registry("GENERATOR")
GENERATOR_REGISTRY.__doc__ = """
Registry for Generators.
The registered object will be called with `obj(cfg, dataset)`.
The call is expected to return an :class:`BaseGenerator`.
"""

logger = logging.getLogger(__name__)


def build_generator(cfg, dataset) -> 'BaseGenerator':
    """
    Build Generators defined by `cfg.GENERATOR.NAME`.
    """
    name = cfg.GENERATOR.NAME
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

    def __len__(self) -> int:
        'Denotes the number of batches per epoch'
        return self.length

    def __getitem__(self,
                    index: int
                    ) -> Tuple[List[Observations], List[Event]]:
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        observations, events = self._data_generation(indexes)
        return observations, events

    def from_config(cls, cfg, dataset):
        raise NotImplementedError

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

    def on_epoch_end(self) -> np.ndarray:
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.size)
        if self.shuffle:
            if self.strict_shuffle:
                raise NotImplementedError
            else:
                np.random.shuffle(self.indexes)
        return self.indexes

    def _filter_telescope_dataset(self, dataset):
        self.src_dataset = dataset
        self.dataset = dataset[dataset[""]]

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


# @GENERATOR_REGISTRY.register()
# class MultiStereoGenerator(BaseGenerator):
#     pass

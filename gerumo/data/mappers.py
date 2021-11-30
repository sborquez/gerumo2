from typing import List
import numpy as np
import pandas as pd
import tables
from fvcore.common.registry import Registry
from ..utils.structures import (
    Task, ReconstructionMode, Event, Observations, Telescope
)
from ..config.config import configurable
from .constants import TELESCOPES

"""
Input Mappers
============
"""

INPUT_MAPPER_REGISTRY = Registry("INPUT_MAPPER_REGISTRY")
INPUT_MAPPER_REGISTRY.__doc__ = """
Registry for Input Mappers.
The registered object will be called with `obj(cfg)`.
The call is expected to return an :class:`InputMapper`.
"""


def build_input_mapper(cfg) -> 'InputMapper':
    """
    Build InputMapper defined by `cfg.INPUT.MAPPER.NAME`.
    """
    name = cfg.INPUT.MAPPER.NAME
    return INPUT_MAPPER_REGISTRY.get(name)(cfg)


class InputMapper:
    """Convert raw input from hdf5 into the model's input format"""

    IMAGE_TABLE = "/dl1/event/telescope/images/tel_{0}"

    CHANNELS_TO_IDX = {
        "image": 3,
        "peak_time": 4,
        "image_mask": 5
    }

    def __init__(self, image_channels: List[str],
                 telescope_features: List[str],
                 mode: ReconstructionMode) -> None:
        self.image_channels = image_channels
        self.telescope_features = telescope_features
        self.mode = mode

    def __call__(self, event_df: pd.DataFrame) -> Observations:
        """Convert event dataframe into Observations structure.
        Args:
            event_df (pd.Dataframe): Dataset of observations for an event.
                If it is SINGLE reconstruction, `event_df` has length 1.
                If it is STEREO reconstruction, `event_df` is all the
                observations of corresponding to an event.
        Returns:
            Event: an event.
        """
        raise NotImplementedError

    def load_image_data(self, hdf5_file: str,
                        tel_id: int, observation_idx: int) -> List[np.ndarray]:
        image_data = []
        h5 = tables.open_file(hdf5_file, "r")
        image_table = InputMapper.IMAGE_TABLE.format(str(tel_id).zfill(3))
        for channel in self.image_channels:
            channel_idx = InputMapper.CHANNELS_TO_IDX[channel]
            image_data.append(
                h5.root[image_table][observation_idx][channel_idx].astype(float)  # noqa
            )
        h5.close()
        return image_data


@INPUT_MAPPER_REGISTRY.register()
class SimpleSquareImage(InputMapper):
    """
    Transform raw format to one square matrix.
    """
    _ARGS = []
    _KWARGS = []

    @configurable
    def __init__(self, image_channels: List[str],
                 telescope_features: List[str],
                 mode: ReconstructionMode) -> None:
        super().__init__(image_channels, telescope_features, mode)

    @classmethod
    def from_config(cls, cfg):
        image_channels = cfg.INPUT.IMAGE_CHANNELS
        telescope_features = cfg.INPUT.TELESCOPE_FEATURES
        reconstruction_mode = cfg.MODEL.RECONSTRUCTION_MODE
        return {
            "image_channels": image_channels,
            "telescope_features": telescope_features,
            "mode": ReconstructionMode[reconstruction_mode]
        }

    def raw_to_image(self, data: List[np.ndarray],
                     telescope: Telescope) -> np.ndarray:
        x, y = telescope.get_aligned_pixels_positions()[0]
        c = len(data)
        w, h = telescope.get_aligned_pixels_positions()[0].max(axis=1) + 1
        image_shape = (h, w, c)
        canvas = np.zeros(image_shape, dtype="float32")
        for i in range(c):
            canvas[y, x, i] = data[i]
        return canvas

    def __call__(self, event_df: pd.DataFrame) -> Observations:
        """Convert event dataframe into Observations structure.
        Args:
            event_df (pd.Dataframe): Dataset of observations for an event.
                If it is SINGLE reconstruction, `event_df` has length 1.
                If it is STEREO reconstruction, `event_df` is all the
                observations of corresponding to an event.
        Returns:
            Observations: observations from an event.
        """
        event_unique_id = event_df.iloc[0].event_unique_id
        telescopes = []
        images = [] if len(self.image_channels) else None
        features = [] if len(self.telescope_features) else None
        for _, event_row in event_df.iterrows():
            telescope = TELESCOPES[event_row.type]
            telescopes.append(telescope)
            if len(self.image_channels) > 0:
                data = self.load_image_data(
                    event_row["hdf5_filepath"],
                    event_row["tel_id"],
                    event_row["observation_idx"]
                )
                images.append(
                    self.raw_to_image(data, telescope)
                )
            if len(self.telescope_features) > 0:
                features.append(
                    event_row[self.telescope_features].values.astype(float)
                )
        observations = Observations(
            event_unique_id=event_unique_id,
            mode=self.mode,
            telescopes=telescopes,
            features_names=self.telescope_features,
            channels_names=self.image_channels,
            images=images,
            features=features
        )
        return observations


"""
Output Mappers
============
"""

OUTPUT_MAPPER_REGISTRY = Registry("OUTPUT_MAPPER_REGISTRY")
OUTPUT_MAPPER_REGISTRY.__doc__ = """
Registry for Output Mappers.
The registered object will be called with `obj(cfg)`.
The call is expected to return an :class:`OutputMapper`.
"""


def build_output_mapper(cfg) -> 'OutputMapper':
    """
    Build OutputMapper defined by `cfg.OUTPUT.MAPPER.NAME`.
    """
    name = cfg.OUTPUT.MAPPER.NAME
    return OUTPUT_MAPPER_REGISTRY.get(name)(cfg)


class OutputMapper:
    """Convert raw input from hdf5 into the model's input format"""
    def __call__(self, event_df: pd.DataFrame) -> Event:
        """Convert event dataframe into Event structure.
        Args:
            event_df (pd.Dataframe): Dataset of observations for an event.
                If it is SINGLE reconstruction, `event_df` has length 1.
                If it is STEREO reconstruction, `event_df` is all the
                observations of corresponding to an event.
        Returns:
            Event: an event.
        """
        raise NotImplementedError


class OutputRegressionMapper(OutputMapper):
    task = Task.REGRESSION

    def __init__(self, targets, domains) -> None:
        super().__init__()
        self.targets = targets
        self.domains = domains

    @classmethod
    def from_config(cls, cfg):
        return {
            "targets": cfg.OUTPUT.REGRESSION.TARGETS,
            "domains": cfg.OUTPUT.REGRESSION.TARGETS_DOMAINS,
        }


@OUTPUT_MAPPER_REGISTRY.register()
class SimpleRegression(OutputRegressionMapper):

    @configurable
    def __init__(self, targets, domains) -> None:
        super().__init__(targets=targets, domains=domains)

    def __call__(self, event_df: pd.DataFrame) -> Event:
        """Convert event dataframe into Event structure for regression.
        Args:
            event_df (pd.Dataframe): Dataset of observations for an event.
                If it is SINGLE reconstruction, `event_df` has length 1.
                If it is STEREO reconstruction, `event_df` is all the
                observations of corresponding to an event.
        Returns:
            Event: an event.
        """
        event = Event.from_dataframe(event_df, self.targets)
        return event


class OutputClassificationMapper(OutputMapper):
    task = Task.CLASSIFICATION

    def __init__(self, target, num_classes, classes) -> None:
        super().__init__()
        assert num_classes == len(classes)
        self.target = target
        self.num_classes = num_classes
        self.classes = classes

    @classmethod
    def from_config(cls, cfg):
        return {
            "target": cfg.INPUT.target,
            "num_classes": cfg.INPUT.NUM_CLASSES,
            "classes": cfg.INPUT.CLASSES
        }


@OUTPUT_MAPPER_REGISTRY.register()
class SimpleCategorical(OutputClassificationMapper):

    @configurable
    def __init__(self, target, num_classes, classes) -> None:
        super().__init__(target, num_classes, classes)

    def __call__(self, event_df: pd.DataFrame) -> Event:
        """Convert event dataframe into Event structure for classification.
        Args:
            event_df (pd.Dataframe): Dataset of observations for an event.
                If it is SINGLE reconstruction, `event_df` has length 1.
                If it is STEREO reconstruction, `event_df` is all the
                observations of corresponding to an event.
        Returns:
            Event: an event.
        """
        event = Event.from_dataframe(event_df, [])
        event.set("true_class_id",
                  self.classes.index(event_df.iloc[0][self.target]))
        return event


@OUTPUT_MAPPER_REGISTRY.register()
class OnevsAllClassification(OutputClassificationMapper):
    @configurable
    def __init__(self, target, num_classes, classes) -> None:
        assert num_classes == 2
        classes = [f"not_{classes[0]}", classes[0]]
        super().__init__(target, num_classes, classes)

    def __call__(self, event_df: pd.DataFrame) -> Event:
        """Convert event dataframe into Event structure for one vs all classification.
        Args:
            event_df (pd.Dataframe): Dataset of observations for an event.
                If it is SINGLE reconstruction, `event_df` has length 1.
                If it is STEREO reconstruction, `event_df` is all the
                observations of corresponding to an event.
        Returns:
            Event: an event.
        """
        event = Event.from_dataframe(event_df, [])
        event.set("true_class_id",
                  int(event_df.iloc[0][self.target] == self.classes[1]))
        return event

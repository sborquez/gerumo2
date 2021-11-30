import pandas as pd
from fvcore.common.registry import Registry
from ..utils.structures import Task, Event, Observations
from ..config.config import configurable

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
    def __call__(self, event_df: pd.DataFrame) -> Observations:
        raise NotImplementedError


@INPUT_MAPPER_REGISTRY.register()
class SimpleSquareImage(InputMapper):

    @configurable
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, event_df: pd.DataFrame) -> Observations:
        pass


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
    def __call__(self, event_df: pd.DataFrame) -> Event:
        raise NotImplementedError


class OutputClassificationMapper(OutputMapper):
    task = Task.CLASSIFICATION

    def __init__(self, target, num_classes, classes) -> None:
        super().__init__()

    @classmethod
    def from_config(cls, cfg):
        return {
            "target": cfg.INPUT.target,
            "num_classes": cfg.INPUT.NUM_CLASSES,
            "classes": cfg.INPUT.CLASSES
        }


class OutputRegressionMapper(OutputMapper):
    task = Task.REGRESSION

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
        pass


@OUTPUT_MAPPER_REGISTRY.register()
class SimpleCategorical(OutputClassificationMapper):
    @configurable
    def __init__(self, target, num_classes, classes) -> None:
        super().__init__(target, num_classes, classes)

    def __call__(self, event_df: pd.DataFrame) -> Event:
        pass


@OUTPUT_MAPPER_REGISTRY.register()
class OnevsAllClassification(OutputClassificationMapper):
    @configurable
    def __init__(self, target, num_classes, classes) -> None:
        super().__init__(target, num_classes, classes)

    def __call__(self, event_df: pd.DataFrame) -> Event:
        pass

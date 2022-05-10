from typing import List, Union, Tuple

import tensorflow as tf

from .base import BaseEnsembler, ENSEMBLER_REGISTRY
from ..utils.structures import Observations


@ENSEMBLER_REGISTRY.register()
class IntensityWeightedAverage(BaseEnsembler):

    def __call__(self, X: List[Observations], uncertainty: bool = True) -> List[Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]]:
        self._hillas_parameter = 'hillas_intensity'
        self._average_weights = self._compute_weights(X)
        return super().__call__(X, uncertainty)

    def _compute_weights(self, X: List[Observations]) -> List[tf.Tensor]:
        weights = []
        for X_i in X:
            hillas_parameters_by_telescope = X_i.hillas_parameters
            hillas_parameters = []
            for telescope in self.telescopes:
                for h in hillas_parameters_by_telescope.get(telescope, []):
                    hillas_parameters.append(
                        h.get(self._hillas_parameter, None)
                    )
                    assert hillas_parameters[-1] is not None, f'InputMapper doesnt include {self._hillas_parameter} into observations, check configuration INPUT.MAPPER.KWARGS'
            weights.append(hillas_parameters)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        weights /= tf.reduce_sum(weights, axis=1, keepdims=True)
        return [weights[i] for i in range(len(weights))]

    def ensemble(self, models_outputs: tf.Tensor) -> tf.Tensor:
        """Combine the outputs for a single event"""
        out_extra_dims = models_outputs.ndim - 1
        weights = tf.reshape(self._average_weights.pop(0), [-1] + [1]*out_extra_dims)
        return tf.reduce_sum(weights * models_outputs)

    def point_estimation(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Convert the ensembler regression output into a point form."""
        return self.models[self.telescopes[0]].point_estimation(y_ensembled)

    def compute_variance(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Variance for regression output."""
        return self.models[self.telescopes[0]].compute_variance(y_ensembled)


@ENSEMBLER_REGISTRY.register()
class Umonne(BaseEnsembler):

    def ensemble(self, models_outputs: tf.Tensor) -> tf.Tensor:
        """Combine the outputs for a single event"""
        return tf.reduce_prod(models_outputs, axis=0, keepdims=True)

    def point_estimation(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Convert the ensembler regression output into a point form."""
        return self.models[self.telescopes[0]].point_estimation(y_ensembled)

    def compute_variance(self, y_ensembled: tf.Tensor) -> tf.Tensor:
        """Variance for regression output."""
        return self.models[self.telescopes[0]].compute_variance(y_ensembled)

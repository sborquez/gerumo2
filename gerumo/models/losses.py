from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from fvcore.common.registry import Registry

from ..config.config import CfgNode

CUSTOM_LOSSES_REGISTRY = Registry("CUSTOM_LOSSES")
CUSTOM_LOSSES_REGISTRY.__doc__ = """
Registry for Losses.
The registered object will be called with `obj(cfg)`.
The call is expected to return an :class:`Loss`.
"""


def build_loss(cfg: CfgNode, output_dim: Optional[np.ndarray] = None) -> Any:
    loss_name = cfg.SOLVER.LOSS.CLASS_NAME
    loss_config = {k: v for k, v in cfg.SOLVER.LOSS.CONFIG}

    if loss_name in CUSTOM_LOSSES_REGISTRY:
        loss_config['target_domains'] = cfg.OUTPUT.REGRESSION.TARGETS_DOMAINS
        loss_config['output_dim'] = output_dim
        loss_ = CUSTOM_LOSSES_REGISTRY.get(loss_name)(**loss_config)
    else:
        loss_ = losses.get({
            'class_name': loss_name,
            'config': loss_config
        })
    return loss_


def tf_ravel_multi_index(multi_index, dims):
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    return tf.reduce_sum(tf.transpose(multi_index) * tf.expand_dims(strides, 1), axis=0)


@CUSTOM_LOSSES_REGISTRY.register
def sparse_nd_crossentropy(output_dim, target_domains, **kwargs):
    output_dim_flat = np.prod(output_dim)
    output_dim = tf.convert_to_tensor(output_dim, dtype='float32')
    target_domains = tf.convert_to_tensor(np.array(target_domains), dtype='float32')
    target_resolutions = (target_domains[:, 1] - target_domains[:, 0])/(output_dim - 1)
    epsilon = tf.keras.backend.epsilon()

    def loss(y_true, y_pred):
        indices_nd = tf.round((y_true - target_domains[:, 0])/target_resolutions)
        indices_1d = tf_ravel_multi_index(indices_nd, output_dim)
        y_pred_1d = tf.reshape(y_pred, (-1, output_dim_flat))
        y_pred_1d = tf.clip_by_value(y_pred_1d, epsilon, 1 - epsilon)
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(indices_1d, y_pred_1d))
    return loss


@CUSTOM_LOSSES_REGISTRY.register
def nd_crossentropy(output_dim, target_domains, **kwargs):
    """Return the Cross Entropy loss function for multidimension probability map."""
    axis = [-i for i in range(1, len(output_dim)+1)]
    epsilon = tf.keras.backend.epsilon()
    output_dim_flat = np.prod(output_dim)
    output_dim = tf.convert_to_tensor(output_dim, dtype='float32')
    target_domains = tf.convert_to_tensor(np.array(target_domains), dtype='float32')
    target_resolutions = (target_domains[:, 1] - target_domains[:, 0])/(output_dim - 1)

    def loss(y_true, y_pred):
        """Cross entropy loss function."""
        # indices_nd = tf.round((y_true - target_domains[:, 0])/target_resolutions)
        # indices_1d = tf_ravel_multi_index(indices_nd, output_dim)
        # y_true_ = tf.reshape(tf.zeros_like(y_pred, dtype='float32'), (-1, output_dim_flat))
        # # y_true_[:, indices_1d] = 1.0
        # y_true_ = tf.reshape(y_true_, y_pred.shape)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        return -tf.reduce_sum(y_pred * tf.math.log(y_pred), axis)
    return loss
import pickle
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn import ensemble
from sklearn import preprocessing

from .base import BaseModel, SKLearnModel, MODEL_REGISTRY
from .layers import (
    HexConvLayer, ConvBlock, PositionalEncoder,
    UpsamplingBlock, DiscreteUncertaintyHead
)
from ..utils.structures import Task, Telescope, InputShape, ReconstructionMode
from ..config.config import configurable


@MODEL_REGISTRY.register()
class CNN(BaseModel):

    _KWARGS = [
        'num_classes', 'num_targets',
        'hexconv', 'conv_kernel_sizes', 'layer_norm', 'last_channels',
        'features_encoding_method', 'features_position', 'dense_layer_units',
        'kernel_regularizer_l1', 'kernel_regularizer_l2',
        'activity_regularizer_l1', 'activity_regularizer_l2'
    ]

    def architecture(
            self,
            num_classes=None, num_targets=None, hexconv=False,
            conv_kernel_sizes=[5, 3, 3], layer_norm=False, last_channels=512,
            features_encoding_method='concat', features_position='first',
            dense_layer_units=[128, 128, 64],
            kernel_regularizer_l1=None, kernel_regularizer_l2=None,
            activity_regularizer_l1=None, activity_regularizer_l2=None):
        # Config validation
        assert num_classes is not None or num_targets is not None
        assert self._input_shape.has_image()
        if self._input_shape.has_features():
            assert features_encoding_method in ('concat', 'positional_encoding')
            assert features_position in ('first', 'last')
            self.features_encoding_method = features_encoding_method
            self._features_position = features_position
        else:
            self.features_encoding_method = None
            self._features_position = None
        # Regularizers
        kl1 = kernel_regularizer_l1
        kl2 = kernel_regularizer_l2
        al1 = activity_regularizer_l1
        al2 = activity_regularizer_l2
        # Inputs Components
        self._input = []
        self._input_img = layers.InputLayer(
            name="images",
            input_shape=self._input_shape.images_shape[1:],
            batch_size=self._input_shape.batch_size
        )
        self._input.append(self._input_img)
        if self._input_shape.has_features():
            self._input_features = layers.InputLayer(
                name="features",
                input_shape=self._input_shape.features_shape[1:],
                batch_size=self._input_shape.batch_size
            )
            self._input.append(self._input_features)
        # Image Branch
        if hexconv:
            self._img_layer = HexConvLayer(filters=64, kernel_size=(3, 3))
        else:
            self._img_layer = layers.Conv2D(
                filters=64, kernel_size=(3, 3), activation="relu",
                kernel_initializer='he_uniform', padding='valid',
                kernel_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                bias_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                activity_regularizer=regularizers.l1_l2(l1=kl1, l2=al2))
        self._conv_blocks = [
            ConvBlock(2**(7+i), k,
                      kernel_regularizer_l1=kl1, kernel_regularizer_l2=kl2,
                      activity_regularizer_l1=kl1, activity_regularizer_l2=al2,
                      layer_norm=layer_norm)
            for i, k in enumerate(conv_kernel_sizes)
        ]
        self._compress_channels = layers.Conv2D(
                filters=last_channels, kernel_size=1, activation="relu",
                kernel_initializer='he_uniform', padding='valid',
                kernel_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                bias_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                activity_regularizer=regularizers.l1_l2(l1=al1, l2=al2)
        )
        self._flatten = layers.Flatten()
        # Feature Branch
        if self._input_shape.has_features():
            if self.features_encoding_method == 'concat':
                self._features_encoder = layers.Concatenate()
            elif self.features_encoding_method == 'positional_encoding':
                # _features_position is (fist|last)
                output_dim_i = 0 if self._features_position == 'first' else -1
                self._features_encoder = PositionalEncoder(
                    input_dim=self._input_shape.features_shape[1],
                    output_dim=dense_layer_units[output_dim_i]
                )
            else:
                ValueError('Invalid "features_encoding_method"', features_encoding_method)  # noqa
        # Logic Components
        self._logic_blocks = [
            (layers.Dense(units), layers.BatchNormalization(), layers.Activation('relu'))  # noqa
            for units in dense_layer_units
        ]
        # Head Components
        if self.task is Task.REGRESSION:
            self.num_classes = None
            self.num_targets = num_targets
            self._head = layers.Dense(num_targets, activation="linear")
        elif self.task is Task.CLASSIFICATION:
            self.num_targets = None
            self.num_classes = num_classes
            self._head = layers.Dense(num_classes, activation="softmax")
        else:
            raise NotImplementedError

    def forward(self, X, training=False):
        # Input
        if self._input_shape.has_features():
            img_X, features_X = X
        else:
            img_X = X
        front = self._input_img(img_X)
        if self._input_shape.has_features():
            front2 = self._input_features(features_X)
        # Encoding
        front = self._img_layer(front)
        for conv_block in self._conv_blocks:
            front = conv_block(front, training=training)
        front = self._compress_channels(front)
        front = self._flatten(front)
        # Logic
        dense, bn, activation = self._logic_blocks[0]
        front = activation(bn(dense(front)))
        if self._input_shape.has_features() \
                and self._features_position == 'first':
            front = self._features_encoder([front, front2])
        for logic_block in self._logic_blocks[1:]:
            dense, bn, activation = logic_block
            front = activation(bn(dense(front)))
        if self._input_shape.has_features() \
                and self._features_position == 'last':
            front = self._features_encoder([front, front2])
        # Head
        return self._head(front)

    def get_output_dim(self):
        if self.task is Task.REGRESSION:
            return np.array([self.num_targets])
        elif self.task is Task.CLASSIFICATION:
            return np.array([self.num_classes])
        else:
            raise NotImplementedError


@MODEL_REGISTRY.register()
class UmonneModel(BaseModel):

    _KWARGS = [
        'num_classes', 'num_targets',
        'hexconv', 'conv_kernel_sizes', 'layer_norm', 'last_channels',
        'features_encoding_method', 'features_position', 'dense_layer_units',
        'upsampling_kernel_sizes',
        'kernel_regularizer_l1', 'kernel_regularizer_l2',
        'activity_regularizer_l1', 'activity_regularizer_l2',
        'point_estimation_mode'
    ]

    @configurable
    def __init__(self, input_shape: InputShape, mode: ReconstructionMode,
                 task: Task, target_domains: List[Tuple] = [],
                 telescopes: List[Telescope] = None,
                 weights: Optional[str] = None,
                 **kwargs):
        self.target_domains = np.array(target_domains)
        super(UmonneModel, self).__init__(
            input_shape, mode, task, telescopes, weights, **kwargs
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        config = super(UmonneModel, cls).from_config(cfg, input_shape)
        config['target_domains'] = cfg.OUTPUT.REGRESSION.TARGETS_DOMAINS
        return config

    def architecture(
            self,
            num_classes=None, num_targets=None, hexconv=False,
            conv_kernel_sizes=[5, 3, 3], layer_norm=True, last_channels=512,
            features_encoding_method='concat', features_position='first',
            dense_layer_units=[128, 128, 64],
            upsampling_kernel_sizes=[3, 3, 3, 3],
            kernel_regularizer_l1=None, kernel_regularizer_l2=None,
            activity_regularizer_l1=None, activity_regularizer_l2=None,
            point_estimation_mode='expected_value'):
        # Config validation
        assert self.task is Task.REGRESSION, 'not supported task'
        assert num_classes is not None or num_targets is not None
        assert self._input_shape.has_image()
        assert len(self.target_domains) == num_targets
        if self._input_shape.has_features():
            assert features_encoding_method in ('concat', 'positional_encoding')
            assert features_position in ('first', 'last')
            self.features_encoding_method = features_encoding_method
            self._features_position = features_position
        else:
            self.features_encoding_method = None
            self._features_position = None
        self.point_estimation_mode = point_estimation_mode
        # Regularizers
        kl1 = kernel_regularizer_l1
        kl2 = kernel_regularizer_l2
        al1 = activity_regularizer_l1
        al2 = activity_regularizer_l2
        # Inputs Components
        self._input = []
        self._input_img = layers.InputLayer(
            name="images",
            input_shape=self._input_shape.images_shape[1:],
            batch_size=self._input_shape.batch_size
        )
        self._input.append(self._input_img)
        if self._input_shape.has_features():
            self._input_features = layers.InputLayer(
                name="features",
                input_shape=self._input_shape.features_shape[1:],
                batch_size=self._input_shape.batch_size
            )
            self._input.append(self._input_features)
        # Image Branch
        if hexconv:
            self._img_layer = HexConvLayer(filters=64, kernel_size=(3, 3))
        else:
            self._img_layer = layers.Conv2D(
                filters=64, kernel_size=(3, 3), activation="relu",
                kernel_initializer='he_uniform', padding='valid',
                kernel_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                bias_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                activity_regularizer=regularizers.l1_l2(l1=al1, l2=al2))
        self._conv_blocks = [
            ConvBlock(2**(7+i), k,
                      kernel_regularizer_l1=kl1, kernel_regularizer_l2=kl2,
                      activity_regularizer_l1=kl1, activity_regularizer_l2=al2,
                      layer_norm=layer_norm)
            for i, k in enumerate(conv_kernel_sizes)
        ]
        self._compress_channels = layers.Conv2D(
                filters=last_channels, kernel_size=1, activation="relu",
                kernel_initializer='he_uniform', padding='valid',
                kernel_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                bias_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                activity_regularizer=regularizers.l1_l2(l1=al1, l2=al2)
        )
        self._flatten = layers.Flatten()
        # Feature Branch
        if self._input_shape.has_features():
            if self.features_encoding_method == 'concat':
                self._features_encoder = layers.Concatenate()
            elif self.features_encoding_method == 'positional_encoding':
                self._features_encoder = None  # layers.Concatenate()
            else:
                ValueError('Invalid "features_encoding_method"', features_encoding_method)  # noqa
        # Logic Components
        self._logic_blocks = [
            (layers.Dense(units), layers.BatchNormalization(), layers.Activation('relu'))  # noqa
            for units in dense_layer_units
        ]
        # Head Components
        if self.task is not Task.REGRESSION:
            raise NotImplementedError
        self.num_classes = None
        self.num_targets = self.output_dim = num_targets
        starting_head_dimensions = self.output_dim*[1] + [-1]
        self._head_reshaper = layers.Reshape(starting_head_dimensions)
        n_upsampling_blocks = len(upsampling_kernel_sizes)
        self._upsampling_blocks = [
            UpsamplingBlock(
                self.output_dim, kernel_size=k,
                filters=2**(1 + n_upsampling_blocks - i))
            for i, k in enumerate(upsampling_kernel_sizes)
        ]
        self._head = DiscreteUncertaintyHead(self.output_dim)
        # Point Estimation
        self.indices = None  # tf.convert_to_tensor(np.indices(axis))

    def forward(self, X, training=False):
        # Input
        if self._input_shape.has_features():
            img_X, features_X = X
        else:
            img_X = X
        front = self._input_img(img_X)
        if self._input_shape.has_features():
            front2 = self._input_features(features_X)
        # Encoding
        front = self._img_layer(front)
        for conv_block in self._conv_blocks:
            front = conv_block(front, training=training)
        front = self._compress_channels(front)
        front = self._flatten(front)
        # Logic
        dense, bn, activation = self._logic_blocks[0]
        front = activation(bn(dense(front)))
        if self._input_shape.has_features() \
                and self._features_position == 'first':
            front = self._features_encoder([front, front2])
        for logic_block in self._logic_blocks[1:]:
            dense, bn, activation = logic_block
            front = activation(bn(dense(front)))
        if self._input_shape.has_features() \
                and self._features_position == 'last':
            front = self._features_encoder([front, front2])
        # Head
        front = self._head_reshaper(front)
        for upsampling_block in self._upsampling_blocks:
            front = upsampling_block(front, training=training)
        return self._head(front)

    def get_output_dim(self):
        return np.product(
            [u._kernel_size for u in self._upsampling_blocks], axis=0
        )

    def point_estimation(self, y):
        """
        Predict points for a batch of predictions `y_predictions` using
        `self.point_estimation_mode` method.
        """
        if self.point_estimation_mode == 'expected_value':
            axis = y.shape[1:]
            if self.indices is None:
                self.indices = tf.convert_to_tensor(np.indices(axis), dtype=float)
            axis_ = 'ijkl'[:len(axis)]
            y_idxs = tf.einsum(f't{axis_},b{axis_}->bt', self.indices, y)
        else:
            y_idxs = None
            raise NotImplementedError(self.point_estimation_mode)
        old = np.array(y.shape[1:]) - 1
        new = self.target_domains
        m = (new[:, 1] - new[:, 0])/old
        b = new[:, 0]
        return (m*y_idxs + b)


@MODEL_REGISTRY.register()
class RandomForest(SKLearnModel):

    _KWARGS = [
        'num_classes',
        'n_estimators',
        'criterion',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'min_weight_fraction_leaf',
        'max_features',
        'max_leaf_nodes',
        'min_impurity_decrease',
        'bootstrap',
        'oob_score',
        'n_jobs',
        'class_weight',
        'ccp_alpha',
        'max_samples',
    ]

    def get_estimator(self,
                      weights=None,
                      num_classes=None,
                      n_estimators=100,
                      criterion="squared_error",  # "gini"
                      max_depth=None, min_samples_split=2, min_samples_leaf=1,
                      min_weight_fraction_leaf=0.0, max_features="auto",
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      bootstrap=True, oob_score=False, n_jobs=None,
                      class_weight=None, ccp_alpha=0.0, max_samples=None
                      ):
        assert self._input_shape.has_features()
        assert not self._input_shape.has_image()
        if weights is not None:
            self.estimator = pickle.load(weights)
            assert \
                (self.task is Task.REGRESSION and isinstance(self.estimator, ensemble.RandomForestRegressor)) or \
                (self.task is Task.CLASSIFICATION and isinstance(self.estimator, ensemble.RandomForestClassifier))

        elif self.task is Task.REGRESSION:
            self.estimator = ensemble.RandomForestRegressor(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            )
        elif self.task is Task.CLASSIFICATION:
            assert num_classes is not None
            self.num_classes = num_classes
            self.encoder = preprocessing.OneHotEncoder(categories=[list(range(num_classes))])
            self.estimator = ensemble.RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    max_features=max_features,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    bootstrap=bootstrap,
                    oob_score=oob_score,
                    n_jobs=n_jobs,
                    class_weight=class_weight,
                    ccp_alpha=ccp_alpha,
                    max_samples=max_samples)
        else:
            raise NotImplementedError

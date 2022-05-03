import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers
from sklearn import ensemble
from sklearn import preprocessing

from .base import BaseModel, SKLearnModel, MODEL_REGISTRY
from .layers import (
    ImageNormalizer, NoDilationImageNormalizer, HexConvLayer, MCForwardPass, ConvBlock, PositionalEncoder,
    UpsamplingBlock, DeltaRegressionHead, DiscreteUncertaintyHead
)
from ..utils.structures import (
    InputShape, Pointing, ReconstructionMode, Task, Telescope, OutputType
)
from ..config.config import configurable
from ..data.constants import TELESCOPE_TIME_PEAK_MAX


@MODEL_REGISTRY.register()
class CNN(BaseModel):

    _KWARGS = [
        'num_classes', 'num_targets',
        'hexconv', 'conv_kernel_sizes', 'conv_channels',
        'layer_norm', 'last_channels',
        'features_encoding_method', 'features_position', 'dense_layer_units',
        'kernel_regularizer_l1', 'kernel_regularizer_l2',
        'activity_regularizer_l1', 'activity_regularizer_l2',
        'normalize_charge', 'time_peak_max', 'apply_mask', 'apply_delta'
    ]

    REGRESSION_OUTPUT_TYPE = OutputType.POINT
    CLASSIFICATION_OUTPUT_TYPE = OutputType.PMF

    def architecture(
            self,
            num_classes=None, num_targets=None, hexconv=False,
            conv_kernel_sizes=[5, 3, 3], conv_channels=[128, 256, 512],
            layer_norm=False, last_channels=256,
            features_encoding_method='concat', features_position='first',
            dense_layer_units=[128, 128],
            kernel_regularizer_l1=None, kernel_regularizer_l2=None,
            activity_regularizer_l1=None, activity_regularizer_l2=None,
            normalize_charge=True, time_peak_max=True, apply_mask=True,
            apply_delta=True):
        # Config validation
        assert len(conv_kernel_sizes) == len(conv_channels), 'Check configuration file.'
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
        self._layer_norm = layer_norm
        # Build architecture components
        self._input_components(normalize_charge, time_peak_max, apply_mask)
        self._image_branch(hexconv,
                           kernel_regularizer_l1, kernel_regularizer_l2,
                           activity_regularizer_l1, activity_regularizer_l2,
                           conv_kernel_sizes, conv_channels, layer_norm,
                           last_channels)
        self._feature_branch(dense_layer_units)
        self._logic_components(dense_layer_units, layer_norm)
        self._head_components(num_targets, num_classes, apply_delta)

    def _input_components(self, normalize_charge, time_peak_max, apply_mask):
        self._input = []
        # Image
        self._input_img = layers.InputLayer(
            name='images',
            input_shape=self._input_shape.images_shape[1:],
            batch_size=self._input_shape.batch_size
        )
        self._input.append(self._input_img)
        if time_peak_max:
            time_peak_max = TELESCOPE_TIME_PEAK_MAX[self.telescopes[0].type]
        else:
            time_peak_max = None
        self._preprocess_image = ImageNormalizer(
            self._input_shape.images_shape,
            normalize_charge, time_peak_max, apply_mask
        )
        # Features
        if self._input_shape.has_features():
            self._input_features = layers.InputLayer(
                name='features',
                input_shape=self._input_shape.features_shape[1:],
                batch_size=self._input_shape.batch_size
            )
            self._input.append(self._input_features)

    def _image_branch(self, hexconv,
                      kernel_regularizer_l1, kernel_regularizer_l2,
                      activity_regularizer_l1, activity_regularizer_l2,
                      conv_kernel_sizes, conv_channels, layer_norm,
                      last_channels):
        if hexconv:
            self._img_layer = HexConvLayer(filters=64, kernel_size=(3, 3))
        else:
            self._img_layer = layers.Conv2D(
                filters=64, kernel_size=(3, 3), activation='relu',
                kernel_initializer='he_uniform', padding='valid',
                kernel_regularizer=regularizers.l1_l2(l1=kernel_regularizer_l1, l2=kernel_regularizer_l2),
                activity_regularizer=regularizers.l1_l2(l1=activity_regularizer_l1, l2=activity_regularizer_l2))
        self._conv_blocks = [
            ConvBlock(filters=f, kernel_size=k,
                      kernel_regularizer_l1=kernel_regularizer_l1, kernel_regularizer_l2=kernel_regularizer_l2,
                      activity_regularizer_l1=activity_regularizer_l1, activity_regularizer_l2=activity_regularizer_l2,
                      layer_norm=layer_norm)
            for k, f in zip(conv_kernel_sizes, conv_channels)
        ]
        self._compress_channels = layers.Conv2D(
            filters=last_channels, kernel_size=1, activation='relu',
            kernel_initializer='he_uniform', padding='valid',
            kernel_regularizer=regularizers.l1_l2(l1=kernel_regularizer_l1, l2=kernel_regularizer_l2),
            bias_regularizer=regularizers.l1_l2(l1=kernel_regularizer_l1, l2=kernel_regularizer_l2),
            activity_regularizer=regularizers.l1_l2(l1=activity_regularizer_l1, l2=activity_regularizer_l2)
        )
        self._flatten = layers.Flatten()
    
    def _feature_branch(self, dense_layer_units):
        self.logic_activations = ['relu'] * len(dense_layer_units)
        if self._input_shape.has_features():
            output_dim_i = 0 if self._features_position == 'first' else -1
            # self.logic_activations[output_dim_i] = 'tanh'
            if self.features_encoding_method == 'concat':
                self._features_encoder = layers.Concatenate()
            elif self.features_encoding_method == 'positional_encoding':
                # _features_position is (fist|last)
                self._features_encoder = PositionalEncoder(
                    input_dim=self._input_shape.features_shape[1],
                    output_dim=dense_layer_units[output_dim_i]
                )
            else:
                ValueError('Invalid "features_encoding_method"', self.features_encoding_method)  # noqa

    def _logic_components(self, dense_layer_units, layer_norm=False):
        self._logic_blocks = [
            (
                layers.Dense(units),
                layers.LayerNormalization() if layer_norm else layers.BatchNormalization(),
                layers.Activation(func)
            )
            for units, func in zip(dense_layer_units, self.logic_activations)
        ]

    def _head_components(self, num_targets, num_classes, apply_delta=False):
        if self.task is Task.REGRESSION:
            self.num_classes = None
            self.num_targets = num_targets
            if apply_delta:
                self._head = DeltaRegressionHead(num_targets, self.pointing)  # NOTE: Use it only with angular
            else:
                self._head = layers.Dense(num_targets, activation='linear')
        elif self.task is Task.CLASSIFICATION:
            self.num_targets = None
            self.num_classes = num_classes
            self._head = layers.Dense(num_classes, activation='softmax')
        else:
            raise NotImplementedError

    def forward(self, X, training=False):
        # Input
        if self._input_shape.has_features():
            img_X, features_X = X
        else:
            img_X = X
        front = self._input_img(img_X)
        front = self._preprocess_image(front)
        if self._input_shape.has_features():
            front2 = self._input_features(features_X)
        # Encoding
        front = self._img_layer(front)
        for conv_block in self._conv_blocks:
            front = conv_block(front, training=training)
        front = self._compress_channels(front)
        front = self._flatten(front)
        # Logic
        dense, norm, activation = self._logic_blocks[0]
        front = dense(front)
        front = norm(front) if self._layer_norm else  norm(front, training=training)
        front = activation(front)
        # Add features at the first layer
        if self._input_shape.has_features() \
                and self._features_position == 'first':
            front = self._features_encoder([front, front2])
        for logic_block in self._logic_blocks[1:]:
            dense, norm, activation = logic_block
            front = dense(front)
            front = norm(front) if self._layer_norm else  norm(front, training=training)
            front = activation(front)
        # Add features at the last layer
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

    def compute_variance(self, y_pred: tf.Tensor):
        batch_size = y_pred.shape[0]
        return tf.reshape((), (batch_size, 0))

    def compute_predictive_entropy(self, y_pred: tf.Tensor):
        batch_size = y_pred.shape[0]
        return tf.reshape((), (batch_size, 0))


@MODEL_REGISTRY.register()
class BMO(CNN):
    _KWARGS = [
        'num_classes', 'num_targets',
        'hexconv', 'conv_kernel_sizes', 'conv_channels',
        'layer_norm', 'last_channels', 'mc_dropout_rate', 'mc_samples',
        'features_encoding_method', 'features_position', 'dense_layer_units',
        'kernel_regularizer_l1', 'kernel_regularizer_l2',
        'activity_regularizer_l1', 'activity_regularizer_l2',
        'normalize_charge', 'time_peak_max', 'apply_mask', 'apply_delta'
    ]

    REGRESSION_OUTPUT_TYPE = OutputType.SAMPLES
    CLASSIFICATION_OUTPUT_TYPE = OutputType.SAMPLES

    def architecture(
            self,
            num_classes=None, num_targets=None, hexconv=False,
            conv_kernel_sizes=[5, 3, 3], conv_channels=[128, 256, 512],
            layer_norm=False, last_channels=256,
            features_encoding_method='concat', features_position='first',
            dense_layer_units=[128, 128], mc_dropout_rate=0.3, mc_samples=100,
            kernel_regularizer_l1=None, kernel_regularizer_l2=None,
            activity_regularizer_l1=None, activity_regularizer_l2=None,
            normalize_charge=True, time_peak_max=True, apply_mask=True,
            apply_delta=True):
        # MC Dropout options
        self.mc_dropout_rate = mc_dropout_rate
        self._mc_samples = mc_samples
        super().architecture(
            num_classes, num_targets, hexconv, conv_kernel_sizes, conv_channels,
            layer_norm, last_channels, features_encoding_method, features_position,
            dense_layer_units, kernel_regularizer_l1, kernel_regularizer_l2,
            activity_regularizer_l1, activity_regularizer_l2,
            normalize_charge, time_peak_max, apply_mask, apply_delta
        )

    def _logic_components(self, dense_layer_units, layer_norm=False):
        self._sampler = MCForwardPass(self._mc_samples)
        self._logic_blocks = [
            (
                layers.Dense(units),
                layers.LayerNormalization() if layer_norm else layers.BatchNormalization(),
                layers.Dropout(rate=self.mc_dropout_rate),
                layers.Activation(func)
            )
            for units, func in zip(dense_layer_units, self.logic_activations)
        ]

    def forward(self, X, training=False):
        # Input
        if self._input_shape.has_features():
            img_X, features_X = X
        else:
            img_X = X
        front = self._input_img(img_X)
        front = self._preprocess_image(front)
        if self._input_shape.has_features():
            front2 = self._input_features(features_X)
        # Encoding
        front = self._img_layer(front)
        for conv_block in self._conv_blocks:
            front = conv_block(front, training=training)
        front = self._compress_channels(front)
        front = self._flatten(front)
        # Logic
        if not (self.enable_fit_mode or training):  # Monte-Carlo Dropout
            front = self._sampler(front)
            if self._input_shape.has_features():
                front2 = self._sampler(front2)
            
        dense, norm, mc_do, activation = self._logic_blocks[0]
        front = dense(front)
        front = norm(front) if self._layer_norm else  norm(front, training=training)
        front = mc_do(front, training=True) # Monte-Carlo Dropout
        front = activation(front)
        # Add features at the first layer
        if self._input_shape.has_features() \
                and self._features_position == 'first':
            front = self._features_encoder([front, front2])
        for logic_block in self._logic_blocks[1:]:
            dense, norm, mc_do, activation = logic_block
            front = dense(front)
            front = norm(front) if self._layer_norm else  norm(front, training=training)
            front = mc_do(front, training=True) # Monte-Carlo Dropout
            front = activation(front)
        # Add features at the last layer
        if self._input_shape.has_features() \
                and self._features_position == 'last':
            front = self._features_encoder([front, front2])
        # Head
        output = self._head(front)
        if not (self.enable_fit_mode or training):  # Monte-Carlo Dropout
            shape = tf.shape(output)
            new_shape = tf.concat([[-1], [self._mc_samples], shape[1:]], axis=0)
            output = tf.reshape(output, new_shape)
        return output

    def point_estimation(self, outputs):
        return tf.reduce_mean(outputs, axis=1)

    def compute_variance(self, y_pred: tf.Tensor):
        """Compute the variance of a Monte-Carlo dropout samples."""
        return tfp.stats.covariance(y_pred, sample_axis=1)

    def compute_predictive_entropy(self, y_pred: tf.Tensor):
        """Compute the predictive entropy of a Monte-Carlo dropout samples."""
        raise NotImplementedError


@MODEL_REGISTRY.register()
class UmonneModel(CNN):

    _KWARGS = [
        # Task
        'num_classes', 'num_targets',
        # Input Image
        'hexconv', 'conv_kernel_sizes', 'conv_channels',
        'layer_norm', 'last_channels',
        # Input Features
        'features_encoding_method', 'features_position', 'dense_layer_units',
        # Regularization
        'kernel_regularizer_l1', 'kernel_regularizer_l2',
        'activity_regularizer_l1', 'activity_regularizer_l2',
        # Image preprocessing
        'normalize_charge', 'time_peak_max', 'apply_mask',
        # Umonne Discrete map
        'upsampling_kernel_sizes', 'point_estimation_mode'
    ]
    
    REGRESSION_OUTPUT_TYPE = OutputType.PMF
    CLASSIFICATION_OUTPUT_TYPE = None

    @configurable
    def __init__(self, input_shape: InputShape, mode: ReconstructionMode,
                 task: Task, target_domains: List[Tuple],
                 telescopes: List[Telescope], pointing: Union[tuple, Pointing],
                 weights: Optional[str] = None,
                 **kwargs):
        self.target_domains = np.array(target_domains)
        super(UmonneModel, self).__init__(
            input_shape, mode, task, telescopes, pointing, weights, **kwargs
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        config = super(UmonneModel, cls).from_config(cfg, input_shape)
        config['target_domains'] = cfg.OUTPUT.REGRESSION.TARGETS_DOMAINS
        return config

    def architecture(
            self,
            num_classes=None, num_targets=None, hexconv=False,
            conv_kernel_sizes=[5, 3, 3], conv_channels=[128, 128, 256],
            layer_norm=True, last_channels=256,
            features_encoding_method='concat', features_position='first',
            dense_layer_units=[128, 128],
            upsampling_kernel_sizes=[3, 3, 3, 3],
            kernel_regularizer_l1=None, kernel_regularizer_l2=None,
            activity_regularizer_l1=None, activity_regularizer_l2=None,
            normalize_charge=True, time_peak_max=True, apply_mask=True,
            point_estimation_mode='expected_value'):
        # Config validation
        assert self.task is Task.REGRESSION, 'not supported task'
        assert len(conv_kernel_sizes) == len(conv_channels), 'Check configuration file.'
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
        self._layer_norm = layer_norm
        self.point_estimation_mode = point_estimation_mode
        # Build architecture components
        self._input_components(normalize_charge, time_peak_max, apply_mask)
        self._image_branch(hexconv,
                           kernel_regularizer_l1, kernel_regularizer_l2,
                           activity_regularizer_l1, activity_regularizer_l2,
                           conv_kernel_sizes, conv_channels, layer_norm,
                           last_channels)
        self._feature_branch(dense_layer_units)
        self._logic_components(dense_layer_units, layer_norm)
        self._head_components(num_targets, num_classes, upsampling_kernel_sizes, layer_norm)

    def _head_components(self, num_targets, num_classes, upsampling_kernel_sizes, layer_norm):
        # Head Components
        if self.task is Task.REGRESSION:
            self.num_classes = None
            self.num_targets = self.output_dim = num_targets
            starting_head_dimensions = self.output_dim * [1] + [-1]
            self._head_reshaper = layers.Reshape(starting_head_dimensions)
            n_upsampling_blocks = len(upsampling_kernel_sizes)
            self._upsampling_blocks = [
                UpsamplingBlock(
                    self.output_dim, kernel_size=k, layer_norm=layer_norm,
                    filters=2**(1 + n_upsampling_blocks - i))
                for i, k in enumerate(upsampling_kernel_sizes)
            ]
            self._head = DiscreteUncertaintyHead(self.output_dim)
            # Point Estimation
            self.indices = None  # tf.convert_to_tensor(np.indices(axis))
        else:
            raise NotImplementedError

    def forward(self, X, training=False):
        # Input
        if self._input_shape.has_features():
            img_X, features_X = X
        else:
            img_X = X
        front = self._input_img(img_X)
        front = self._preprocess_image(front)
        if self._input_shape.has_features():
            front2 = self._input_features(features_X)
        # Encoding
        front = self._img_layer(front)
        for conv_block in self._conv_blocks:
            front = conv_block(front, training=training)
        front = self._compress_channels(front)
        front = self._flatten(front)
        # Logic
        dense, norm, activation = self._logic_blocks[0]
        front = dense(front)
        front = norm(front) if self._layer_norm else  norm(front, training=training)
        front = activation(front)
        # Add features at the first layer
        if self._input_shape.has_features() \
                and self._features_position == 'first':
            front = self._features_encoder([front, front2])
        for logic_block in self._logic_blocks[1:]:
            dense, norm, activation = logic_block
            front = dense(front)
            front = norm(front) if self._layer_norm else  norm(front, training=training)
            front = activation(front)
        # Add features at the last layer
        if self._input_shape.has_features() \
                and self._features_position == 'last':
            front = self._features_encoder([front, front2])
        # Head
        front = self._head_reshaper(front)
        for upsampling_block in self._upsampling_blocks:
            front = upsampling_block(front, training=training)
        output = self._head(front)
        return output

    def get_output_dim(self):
        return np.product(
            [u._kernel_size for u in self._upsampling_blocks], axis=0
        )

    def point_estimation(self, outputs):
        """
        Predict points for a batch of predictions `outputs_predictions` using
        `self.point_estimation_mode` method.
        """
        if self.point_estimation_mode == 'expected_value':
            axis = outputs.shape[1:]
            if self.indices is None:
                self.indices = tf.convert_to_tensor(np.indices(axis), dtype=float)
            axis_ = 'ijkl'[:len(axis)]
            outputs_idxs = tf.einsum(f't{axis_},b{axis_}->bt', self.indices, outputs)
        else:
            outputs_idxs = None
            raise NotImplementedError(self.point_estimation_mode)
        old = np.array(outputs.shape[1:]) - 1
        new = self.target_domains
        m = (new[:, 1] - new[:, 0]) / old
        b = new[:, 0]
        return (m * outputs_idxs + b)
    
    def compute_variance(self, y_pred: tf.Tensor):
        """Compute the variance of a discrete probability map."""
        batch_size, *dims = y_pred.shape
        assert len(dims ) == 2
        domains = self.target_domains
        axis = 'xyz'[:len(dims)]
        mus = []
        fs = []
        ps = []
        corrs = []
        for i, (dim_i, domain_i) in enumerate(zip(dims, domains)):
            f_i = tf.convert_to_tensor(np.linspace(*domain_i, num=dim_i, dtype=np.float32))
            p_i = tf.einsum(f'b{axis}->b{axis[i]}', y_pred)
            m_i =  tf.einsum('bi,i->b', p_i, f_i)
            corr_i = tf.einsum('bi,i->b', p_i, np.power(f_i, 2)) - np.power(m_i, 2)
            fs.append(f_i)
            ps.append(p_i)
            mus.append(m_i)
            corrs.append(corr_i)

        y_pred_p = np.copy(y_pred)
        for i, f_i in enumerate(fs):
            broadcast_shape = [1]*(1 + len(dims))
            broadcast_shape[i+1] = -1
            y_pred_p *= tf.reshape(f_i, broadcast_shape)
        exp_prod = np.sum(y_pred_p, axis=(1,2))    
        cov = exp_prod - np.prod(mus, axis=0).flatten()

        u = np.zeros((batch_size, 2, 2))
        u[:, 0, 0] = corrs[0]
        u[:, 1, 1] = corrs[1]
        u[:, 0, 1] = cov
        u[:, 1, 0] = cov
        return tf.convert_to_tensor(u)
    
    def compute_predictive_entropy(self, y_pred: tf.Tensor):
        """Compute the predictive entropy of a Monte-Carlo dropout samples."""
        raise NotImplementedError


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
    
    REGRESSION_OUTPUT_TYPE = OutputType.POINT
    CLASSIFICATION_OUTPUT_TYPE = OutputType.POINT

    def get_estimator(self,
                      weights=None,
                      num_classes=None,
                      n_estimators=100,
                      criterion='squared_error',  # 'gini'
                      max_depth=None, min_samples_split=2, min_samples_leaf=1,
                      min_weight_fraction_leaf=0.0, max_features='auto',
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
                max_samples=max_samples
            )
        else:
            raise NotImplementedError


@MODEL_REGISTRY.register()
class OnionCNN(BaseModel):

    _KWARGS = [
        'num_classes', 'num_targets',
        'hexconv', 'conv_kernel_sizes', 'layer_norm', 'last_channels',
        'features_encoding_method', 'features_position', 'dense_layer_units',
        'kernel_regularizer_l1', 'kernel_regularizer_l2',
        'activity_regularizer_l1', 'activity_regularizer_l2',
        'normalize_charge', 'time_peak_max', 'apply_mask'
    ]

    REGRESSION_OUTPUT_TYPE = OutputType.POINT
    CLASSIFICATION_OUTPUT_TYPE = OutputType.PMF

    def architecture(
            self,
            num_classes=None, num_targets=None, hexconv=False,
            conv_kernel_sizes=[5, 3, 3], layer_norm=False, last_channels=512,
            features_encoding_method='concat', features_position='first',
            dense_layer_units=[128, 128, 64],
            kernel_regularizer_l1=None, kernel_regularizer_l2=None,
            activity_regularizer_l1=None, activity_regularizer_l2=None,
            normalize_charge=False, time_peak_max=None, apply_mask=False):
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
        self._input_img = layers.InputLayer(
            name="images",
            input_shape=self._input_shape.images_shape[1:],
            batch_size=self._input_shape.batch_size
        )
        # Image Branch
        self.encoder = tf.keras.Sequential([
            self._input_img,
            NoDilationImageNormalizer(self._input_shape.images_shape, normalize_charge, time_peak_max, apply_mask)
        ])
        if hexconv:
            self.encoder.add(HexConvLayer(filters=64, kernel_size=(3, 3)))
        else:
            self.encoder.add(
                layers.Conv2D(
                    filters=64, kernel_size=(3, 3), activation="relu",
                    kernel_initializer='he_uniform', padding='valid',
                    kernel_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                    bias_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
                    activity_regularizer=regularizers.l1_l2(l1=kl1, l2=al2)
                )
            )
          
        for i, k in enumerate(conv_kernel_sizes):
            self.encoder.add(
                ConvBlock(2**(7 + i), k,
                      kernel_regularizer_l1=kl1, kernel_regularizer_l2=kl2,
                      activity_regularizer_l1=kl1, activity_regularizer_l2=al2,
                      layer_norm=layer_norm)
            )
        
        self.encoder.add(layers.Conv2D(
            filters=last_channels, kernel_size=1, activation="relu",
            kernel_initializer='he_uniform', padding='valid',
            kernel_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
            bias_regularizer=regularizers.l1_l2(l1=kl1, l2=kl2),
            activity_regularizer=regularizers.l1_l2(l1=al1, l2=al2)
        ))
            
        self.encoder.add(layers.Flatten())
        
        # Logic Components
        last_model = self.encoder
        self._logic_blocks = []
        
        for units in dense_layer_units:
            self._logic_blocks.append(
                tf.keras.Sequential([
                    last_model,
                    layers.Dense(units),
                    layers.BatchNormalization(),
                    layers.Activation('relu')  
                ])
            )
            last_model = self._logic_blocks[-1]
        # Head Components
        if self.task is Task.CLASSIFICATION:
            self.num_targets = None
            self.num_classes = num_classes
            self._head = tf.keras.Sequential([
                last_model,
                layers.Dense(num_classes, activation="softmax")
            ])
        else:
            raise NotImplementedError

    def forward(self, X, training=False):
        # Input
        img_X = X
        front = self._input_img(img_X)
        return self._head(front, training=training)

    def get_output_dim(self):
        if self.task is Task.REGRESSION:
            return np.array([self.num_targets])
        elif self.task is Task.CLASSIFICATION:
            return np.array([self.num_classes])
        else:
            raise NotImplementedError

    def compute_predictive_entropy(self, y_pred: tf.Tensor):
        batch_size = y_pred.shape[0]
        return tf.reshape((), (batch_size, 0))

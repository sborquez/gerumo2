from copyreg import pickle
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn import ensemble
from sklearn import preprocessing

from .base import BaseModel, SKLearnModel, MODEL_REGISTRY
from .layers import HexConvLayer, ConvBlock
from ..utils.structures import Task


@MODEL_REGISTRY.register()
class CNN(BaseModel):

    _KWARGS = [
        'num_classes', 'num_targets',
        'hexconv', 'conv_kernel_sizes', 'last_channels',
        'dense_layer_units', 'kernel_regularizer_l2', 'activity_regularizer_l1'
    ]

    def architecture(self,
                     num_classes=None, num_targets=None, hexconv=False,
                     conv_kernel_sizes=[5, 3, 3], last_channels=512,
                     dense_layer_units=[128, 128, 64],
                     kernel_regularizer_l2=None, activity_regularizer_l1=None):
        assert num_classes is not None or num_targets is not None
        assert self._input_shape.has_image()
        l2 = kernel_regularizer_l2
        l1 = activity_regularizer_l1
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
                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                bias_regularizer=regularizers.l2(l2),
                activity_regularizer=regularizers.l1(l1))
        self._img_dropout = layers.Dropout(0.25)
        self._conv_blocks = [
            ConvBlock(2**(7+i), k, l1=1e-5, l2=1e-4)
            for i, k in enumerate(conv_kernel_sizes)
        ]
        self._compress_channels = layers.Conv2D(
                filters=last_channels, kernel_size=1, activation="relu",
                kernel_initializer='he_uniform', padding='valid',
                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                bias_regularizer=regularizers.l2(l2),
                activity_regularizer=regularizers.l1(l1)
        )
        self._flatten = layers.Flatten()
        # Concat Branch
        if self._input_shape.has_features():
            self._concat = layers.Concatenate()
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
        if self._input_shape.has_features():
            img_X, features_X = X
        else:
            img_X = X
        front = self._input_img(img_X)
        front = self._img_layer(front)
        front = self._img_dropout(front, training=training)
        for conv_block in self._conv_blocks:
            front = conv_block(front, training=training)
        front = self._compress_channels(front)
        front = self._flatten(front)
        if self._input_shape.has_features():
            front2 = self._input_features(features_X)
            front = self._concat([front, front2])
        for logic_block in self._logic_blocks:
            dense, bn, activation = logic_block
            front = dense(front)
            front = bn(front)
            front = activation(front)
        return self._head(front)


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

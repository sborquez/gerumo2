from tensorflow.keras import layers
from tensorflow.keras import regularizers
from .base import BaseModel, MODEL_REGISTRY
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

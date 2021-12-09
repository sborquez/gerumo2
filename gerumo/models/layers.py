"""
Custom Layers
=============

Collection of models custom layers.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def softmax(target, axis, name=None):
    """
    Adapted from https://gist.github.com/raingo/a5808fe356b8da031837

    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    max_axis = tf.reduce_max(target, axis, keepdims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
    softmax = target_exp / normalize
    return softmax


class HexConvLayer(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), name=None, reshape=None,
                 **kargs):
        super(HexConvLayer, self).__init__(name=name, **kargs)
        self.__name = name
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv2d = layers.Conv2D(
            filters, kernel_size,
            kernel_initializer="he_uniform",
            padding="valid", activation="relu",
            strides=(2, 1))
        self.left_split = layers.Lambda(lambda x: x[:, 0, :-1, :, :])
        self.right_split = layers.Lambda(lambda x: x[:, 1, 1:, :, :])
        self.concatenate = layers.Concatenate(axis=2)
        self.pooling = layers.MaxPooling2D(pool_size=(2, 2))
        self.__reshape = reshape
        if self.__reshape is not None:
            self.reshape = layers.Reshape(self.__reshape)
        # Infer shape in running time
        else:
            self.reshape = None

    # @tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        input_left = self.left_split(inputs)
        input_right = self.right_split(inputs)
        front_left = self.conv2d(input_left)
        front_right = self.conv2d(input_right)
        front = self.concatenate([front_left, front_right])

        # Intercalate on axis Y the application of the convolution.
        cshape = front.get_shape()
        if self.reshape is None:
            self.__reshape = (
                int(cshape[1])*2, int(cshape[2])//2, int(cshape[3])
                )
            self.reshape = layers.Reshape(self.__reshape)
        front = self.reshape(front)

        return self.pooling(front)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "name": self.__name,
            "reshape": self.__reshape
        })
        return config


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, l1=1e-5, l2=1e-4, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        # Config
        self._kernel_size = kernel_size
        self._filters = filters
        self._l1 = l1
        self._l2 = l2
        # Layers
        self._conv1 = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same',
            kernel_initializer="he_uniform",
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l2(l2),
            activity_regularizer=regularizers.l1(l1)
        )
        self._bn1 = layers.BatchNormalization()
        self._activation1 = layers.Activation('relu')
        self._drop1 = layers.Dropout(0.25)
        self._conv2 = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same',
            kernel_initializer="he_uniform",
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l2(l2),
            activity_regularizer=regularizers.l1(l1)
        )
        self._bn2 = layers.BatchNormalization()
        self._activation2 = layers.Activation('relu')
        self._maxpool = layers.MaxPool2D()

    def call(self, inputs, training=False):
        x = self._conv1(inputs)
        x = self._bn1(x)
        x = self._activation1(x)
        x = self._drop1(x, training=training)
        x = self._conv2(x)
        x = self._bn2(x)
        x = self._activation2(x)
        x = self._maxpool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self._kernel_size,
            'filters': self._filters,
            'l1': self._l1,
            'l2': self._l2
        })
        return config

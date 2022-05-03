"""
Custom Layers
=============

Collection of models custom layers.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from skimage.draw import disk


class ImageNormalizer(layers.Layer):
    def __init__(self, img_shape, normalize_charge=True, time_peak_max=None, apply_mask=True, **kargs):
        assert img_shape[-1] == 3, 'Only works for 3 channels image (charge, time, masks)'
        super(ImageNormalizer, self).__init__(**kargs)
        # Image shape
        self.img_shape = img_shape
        # Apply Layer normalization to charge
        self.charge_split = layers.Lambda(
            lambda x: tf.reshape(x[:, :, :, 0], (img_shape[0], img_shape[1] * img_shape[2]))
        )
        self.normalize_charge = normalize_charge
        if self.normalize_charge:
            self.charge_normalization = layers.LayerNormalization(axis=[1])
        # Apply max scaling to time peak
        self.time_peak_max = time_peak_max
        self.time_peak_split = layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, 1], axis=-1))
        if self.time_peak_max:
            self.time_peak_normalization = layers.Lambda(lambda x: x / self.time_peak_max)

        self.apply_mask = apply_mask
        # Apply masking to the others channels using the mask channel
        self.mask_split = layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, 2], axis=-1))
        self.kernel = self._get_dilation_kernel(size=7)
        self.mask_dilation = layers.Lambda(
            lambda x: tf.nn.dilation2d(x, self.kernel, strides=(1,1,1,1), dilations=(1,1,1,1), padding="SAME", data_format='NHWC') - tf.ones_like(x)
        )
        if self.apply_mask:
            self.masking = layers.Lambda(lambda x: x[0] * x[1])
        else:
            self.concatenate_mask = layers.Concatenate(axis=-1)
        # Concatenate the channels into a single output image
        self.concatenate = layers.Concatenate(axis=-1)

    def _get_dilation_kernel(self, size):
        kernel = np.zeros((size, size, 1), dtype=np.float32)
        rr, cc = disk((size//2, size//2), size/2)
        kernel[rr, cc] = 1
        return tf.constant(kernel)

    def call(self, inputs):
        # Split channels
        charge = self.charge_split(inputs)
        time = self.time_peak_split(inputs)
        mask = self.mask_split(inputs)
        mask = self.mask_dilation(mask)
        # Apply layer normalization
        if self.normalize_charge:
            charge = self.charge_normalization(charge)
        charge = tf.reshape(charge, (self.img_shape[0], self.img_shape[1], self.img_shape[2], 1))
        # Apply Max scaling
        if self.time_peak_max is not None:
            time = self.time_peak_normalization(time)
        # Concatenate charge and time peaks
        output = self.concatenate([charge, time])
        # Apply or concatenate mask
        if self.apply_mask:
            output = self.masking((output, mask))
        else:
            output = self.concatenate_mask([output, mask])
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'img_shape': self.img_shape,
            'normalize_charge': self.normalize_charge,
            'time_peak_max': self.time_peak_max,
            'apply_mask': self.apply_mask
        })
        return config


class NoDilationImageNormalizer(layers.Layer):
    # Asume que la imagen siempre tiene 3 canales ordenados
    def __init__(self, img_shape, normalize_charge=False, time_peak_max=None, apply_mask=False, **kargs):
        super(NoDilationImageNormalizer, self).__init__(**kargs)
        self.img_shape = img_shape
        self.normalize_charge = normalize_charge
        self.time_peak_max = time_peak_max
        self.apply_mask = apply_mask
        self.charge_split = layers.Lambda(lambda x: tf.reshape(x[:, :, :, 0],(img_shape[0],img_shape[1]*img_shape[2])))
        self.time_peak_split = layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, 1], axis = -1))
        self.mask_split = layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, 2], axis = -1))
        self.concatenate = layers.Concatenate(axis=-1)
        if self.normalize_charge:     
            self.charge_normalization = layers.LayerNormalization(axis=-1)
        if self.time_peak_max is not None:
            self.time_peak_normalization = layers.Lambda(lambda x: x/self.time_peak_max)
        if self.apply_mask:
            self.masking = layers.Lambda(lambda x: x[0]*x[1])
        else:
            self.concatenate_mask = layers.Concatenate(axis=-1)

    def call(self, inputs):
        charge = self.charge_split(inputs)
        time = self.time_peak_split(inputs)
        mask = self.mask_split(inputs)
        if self.normalize_charge:
            charge = self.charge_normalization(charge)
        charge = tf.reshape(charge,(self.img_shape[0],self.img_shape[1],self.img_shape[2],1))
        if self.time_peak_max is not None:
            time = self.time_peak_normalization(time)
        output = self.concatenate([charge, time]) 
        if self.apply_mask:
            output = self.masking((output, mask))
        else:
            output = self.concatenate_mask([output, mask])

        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'img_shape': self.img_shape,
            'normalize_charge': self.normalize_charge,
            'time_peak_max': self.time_peak_max,
            'apply_mask': self.apply_mask            
        })
        return config


class HexConvLayer(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), name=None, reshape=None,
                 **kargs):
        super(HexConvLayer, self).__init__(name=name, **kargs)
        self.__name = name
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv2d = layers.Conv2D(
            filters, kernel_size,
            kernel_initializer='he_uniform',
            padding='valid', activation='relu',
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
                int(cshape[1]) * 2, int(cshape[2]) // 2, int(cshape[3])
            )
            self.reshape = layers.Reshape(self.__reshape)
        front = self.reshape(front)

        return self.pooling(front)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'name': self.__name,
            'reshape': self.__reshape
        })
        return config


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size,
                 kernel_regularizer_l1=None, kernel_regularizer_l2=1e-4,
                 activity_regularizer_l1=1e-5, activity_regularizer_l2=None,
                 layer_norm=False, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        # Config
        self._kernel_size = kernel_size
        self._filters = filters
        self._kernel_regularizer_l1 = kernel_regularizer_l1
        self._kernel_regularizer_l2 = kernel_regularizer_l2
        self._activity_regularizer_l1 = activity_regularizer_l1
        self._activity_regularizer_l2 = activity_regularizer_l2
        self._layer_norm = layer_norm
        # Layers
        self._conv1 = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same',
            kernel_initializer='he_uniform',
            kernel_regularizer=regularizers.l1_l2(
                l1=kernel_regularizer_l1, l2=kernel_regularizer_l2),
            bias_regularizer=regularizers.l1_l2(
                l1=kernel_regularizer_l1, l2=kernel_regularizer_l2),
            activity_regularizer=regularizers.l1_l2(
                l1=activity_regularizer_l1, l2=activity_regularizer_l2)
        )
        self._activation1 = layers.Activation('relu')
        self._drop = layers.Dropout(0.25)
        self._conv2 = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, padding='same',
            kernel_initializer='he_uniform',
            kernel_regularizer=regularizers.l1_l2(
                l1=kernel_regularizer_l1, l2=kernel_regularizer_l2),
            bias_regularizer=regularizers.l1_l2(
                l1=kernel_regularizer_l1, l2=kernel_regularizer_l2),
            activity_regularizer=regularizers.l1_l2(
                l1=activity_regularizer_l1, l2=activity_regularizer_l2)
        )
        if layer_norm:
            self._normalization = layers.LayerNormalization(axis=(1,2))
        else:
            self._normalization = layers.BatchNormalization()
        self._activation2 = layers.Activation('relu')
        self._maxpool = layers.MaxPool2D()

    def call(self, inputs, training=False):
        x = self._conv1(inputs)
        x = self._activation1(x)
        x = self._drop(x, training=training)
        x = self._conv2(x)
        x = self._normalization(x) if self._layer_norm else self._normalization(x, training=training)
        x = self._activation2(x)
        x = self._maxpool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self._kernel_size,
            'filters': self._filters,
            'activity_regularizer_l1': self._activity_regularizer_l1,
            'activity_regularizer_l2': self._activity_regularizer_l2,
            'kernel_regularizer_l1': self._kernel_regularizer_l1,
            'kernel_regularizer_l2': self._kernel_regularizer_l2,
            'layer_norm': self._layer_norm
        })
        return config


class SinusoidalEncodingNDim(layers.Layer):
    def __init__(self, input_dim, output_dim, const=10000, **kwargs):
        super(SinusoidalEncodingNDim, self).__init__(**kwargs)
        assert output_dim % (2 * input_dim) == 0
        # Config
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._const = const
        # Layers
        indices = (2 * input_dim) * np.arange(0, np.ceil(output_dim / (2 * input_dim))) / output_dim
        self.inv_freq = np.power(const, -1 * indices)

    def call(self, inputs, training=False):
        freq = tf.einsum('ij,k->ijk', inputs, self.inv_freq)
        sin = tf.math.sin(freq)
        cos = tf.math.cos(freq)
        stack = tf.stack([sin, cos], axis=-1)
        encoding = tf.reshape(stack, [-1, self._output_dim])
        return encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'const': self._const
        })
        return config


class PositionalEncoder(layers.Layer):
    def __init__(self, input_dim, output_dim, const=10000, **kwargs):
        super(PositionalEncoder, self).__init__(**kwargs)
        assert output_dim % (2 * input_dim) == 0
        # Config
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._const = const
        # Layers
        self.sinusolidal_encoder = SinusoidalEncodingNDim(
            input_dim, output_dim, const, **kwargs)
        self.adder = layers.Add()

    def call(self, inputs, training=False):
        # inputs[0]: image features
        # inputs[1]: features to concatenate
        encoding = self.sinusolidal_encoder(inputs[1])
        return self.adder([encoding, inputs[0]])

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'const': self._const
        })
        return config


class UpsamplingBlock(layers.Layer):
    def __init__(self, output_dim, kernel_size, filters, layer_norm, **kwargs):
        super(UpsamplingBlock, self).__init__(**kwargs)
        # Config
        self._output_dim = output_dim
        if output_dim > 1 and isinstance(kernel_size, int):
            kernel_size = tuple(output_dim * [kernel_size])
        self._kernel_size = kernel_size
        self._filters = filters
        self._layer_norm = layer_norm
        # Layer builders
        if output_dim == 1:
            upsampling = layers.UpSampling1D
            conv = layers.Conv1D
            average = layers.AveragePooling1D
            axis = 1
        elif output_dim == 2:
            upsampling = layers.UpSampling2D
            conv = layers.Conv2D
            average = layers.AveragePooling2D
            axis = (1, 2)
        elif output_dim == 3:
            upsampling = layers.UpSampling3D
            conv = layers.Conv3D
            average = layers.AveragePooling3D
            axis = (1, 2, 3)
        else:
            raise ValueError('output_dim should be in [1,3]')
        # Layers
        self.upsampling = upsampling(size=kernel_size)
        self.average = average(kernel_size, 1, 'same')
        self.conv1 = conv(filters, kernel_size, 1, 'same', activation='relu')
        self.norm1 = layers.LayerNormalization(axis=axis) if layer_norm else layers.BatchNormalization()
        self.conv2 = conv(filters, kernel_size, 1, 'same', activation='relu')
        self.norm2 = layers.LayerNormalization(axis=axis) if layer_norm else layers.BatchNormalization()

    def call(self, inputs, training=False):
        front = self.upsampling(inputs)
        front = self.average(front)
        front = self.conv1(front)
        front = self.norm1(front) if self._layer_norm else self.norm1(front, training=training)
        front = self.conv2(front)
        front = self.norm2(front) if self._layer_norm else self.norm2(front, training=training)
        return front

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self._filters,
            'output_dim': self._output_dim,
            'kernel_size': self._kernel_size,
            'layer_norm': self._layer_norm
        })
        return config


class MCForwardPass(layers.Layer):
    def __init__(self, mc_samples, **kwargs):
        """Generate T MC forward passes
        """
        super(MCForwardPass, self).__init__(**kwargs)
        # Config
        self._mc_samples = mc_samples
        self._sampler = layers.RepeatVector(self._mc_samples)

    def call(self, inputs, training=False):
        shape = tf.shape(inputs)
        front = self._sampler(inputs)
        new_shape = tf.concat([shape[:1] * self._mc_samples, shape[1:]], axis=0)
        front = tf.reshape(front, new_shape)
        return front

    def get_config(self):
        config = super().get_config()
        config.update({
            'mc_samples': self._mc_samples,
        })
        return config


class DeltaRegressionHead(layers.Layer):
    def __init__(self, num_targets, offsets, **kwargs):
        super(DeltaRegressionHead, self).__init__(**kwargs)
        self.num_targets = num_targets
        self.offsets = offsets
        self._offsets = tf.constant(offsets, shape=(1, num_targets), dtype=tf.float32)
        self.compute_deltas = layers.Dense(num_targets, activation='linear')

    def call(self, inputs, training=False):
        deltas = self.compute_deltas(inputs)
        return tf.math.add(deltas, self._offsets)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_targets': self.num_targets,
            'offsets': self.offsets
        })
        return config


class DiscreteUncertaintyHead(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        """Generate a discrete distribution from n-axis tensor
        """
        super(DiscreteUncertaintyHead, self).__init__(**kwargs)
        # Config
        self._output_dim = output_dim
        # Layer builders
        if output_dim == 1:
            self.axis = [1]
            conv = layers.Conv1D
        elif output_dim == 2:
            self.axis = [1, 2]
            conv = layers.Conv2D
        elif output_dim == 3:
            self.axis = [1, 2, 3]
            conv = layers.Conv3D
        else:
            raise ValueError('output_dim should be in [1,3]')
        # Layers
        self.conv = conv(1, 1, strides=1, padding='same')
        self.softmax = tf.keras.layers.Softmax(axis=self.axis)
        
    def call(self, inputs, training=False):
        front = self.conv(inputs)
        front = tf.squeeze(front, axis=[-1])
        front = self.softmax(front)
        return front

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self._output_dim,
        })
        return config

import tensorflow as tf
from tensorflow.keras import layers


class SqueezeAndExcitation(tf.keras.layers.Layer):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SqueezeAndExcitation, self).__init__()
        self.input_channels = input_channels
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(
            input_channels // reduction_ratio,
            activation='relu',
            kernel_initializer='he_normal'
        )
        self.dense2 = layers.Dense(
            input_channels,
            activation='sigmoid',
            kernel_initializer='he_normal'
        )

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = tf.reshape(x, [-1, 1, 1, self.input_channels])
        x = self.dense1(x)
        x = self.dense2(x)
        return inputs * x

    def get_config(self):
        config = {
            'input_channels': self.input_channels,
            'reduction_ratio': self.reduction_ratio
        }
        base_config = super(SqueezeAndExcitation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

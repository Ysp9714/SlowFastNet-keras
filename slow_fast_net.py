import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D
import numpy as np


def swish(x, beta=1):
    return x * tf.nn.sigmoid(beta * x)


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


tf.keras.utils.get_custom_objects().update({"swish": swish})
tf.keras.utils.get_custom_objects().update({"mish": mish})


def auto_pad(inputs, kernel_size, data_format):

    """
    This function replaces the padding implementation in original tensorflow.
    It also avoids negative dimension by automatically padding given the input kernel size (for each dimension).
    """

    islist = isinstance(kernel_size, tuple)

    kernel_size = np.array(kernel_size)
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if islist:
        paddings = np.concatenate(
            [pad_beg[:, np.newaxis], pad_end[:, np.newaxis]], axis=1
        )
        paddings = [list(p) for p in paddings]
    else:
        paddings = [[pad_beg, pad_end]] * 3

    if data_format == "channels_first":
        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [0, 0]] + paddings)
    else:
        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0]] + paddings + [[0, 0]])
    return padded_inputs


class ConvXD(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding="valid",
        use_bias=False,
        name="conv_3d",
        data_format="channels_last",
        **kwargs
    ):
        super(ConvXD, self).__init__(name=name, **kwargs)
        # self.name = name
        self.pad = False
        if isinstance(strides, list) or isinstance(strides, tuple):
            self.pad = any(np.array(strides) > 1)
        else:
            self.pad = strides > 1

        if self.pad:
            padding = "valid"
        else:
            padding = "same"

        self.conv = Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
        )

        self.data_format = data_format
        self.kernel_size = kernel_size

    @tf.function
    def call(self, inputs, training=None):
        if self.pad:
            inputs = auto_pad(
                inputs=inputs,
                kernel_size=self.kernel_size,
                data_format=self.data_format,
            )

        outputs = self.conv(inputs)
        return outputs


class PreactBlock(tf.keras.layers.Layer):
    def __init__(self, activation="swish", **kwargs):
        super(PreactBlock, self).__init__()
        self.batch_norm = layers.BatchNormalization(epsilon=1e-5)
        self.act = layers.Activation(activation)

    def get_config(self):
        return super().get_config()

    @tf.function
    def call(self, inputs, training=False):
        x = self.batch_norm(inputs, training)
        x = self.act(x)
        return x


class DataLayer(layers.Layer):
    def __init__(self, stride):
        super(DataLayer, self).__init__()
        self.stride = stride

    @tf.function
    def call(self, inputs):
        x = inputs[:, :: self.stride, :, :, :]
        return x

    def get_config(self):
        config = super(DataLayer, self).get_config()
        config.update({"stride": self.stride})
        return config


class InitBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, use_bias=False):
        super(InitBlock, self).__init__()
        self.conv3d = ConvXD(
            filters, kernel_size, strides=(1, 2, 2), padding="same", use_bias=use_bias
        )
        self.maxpl3d = layers.MaxPool3D(
            pool_size=(1, 3, 3), strides=(1, 2, 2), padding="same"
        )

    def get_config(self):
        return super().get_config()

    @tf.function
    def call(self, inputs, training=False):
        x = self.conv3d(inputs)
        x = self.maxpl3d(x)
        return x


class ResBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        time_kernel_size=1,
        stride=1,
        shortcut=None,
        shortcut_stride=1,
        use_bias=False,
        **kwargs
    ):
        super(ResBlock, self).__init__()
        self.conv3d1 = ConvXD(
            filters, (time_kernel_size, 1, 1), 1, padding="same", use_bias=use_bias
        )
        self.preact1 = PreactBlock()

        self.conv3d2 = ConvXD(
            filters, (1, 3, 3), (1, stride, stride), padding="same", use_bias=use_bias
        )
        self.preact2 = PreactBlock()

        self.conv3d3 = ConvXD(
            filters * 4, 1, strides=1, padding="same", use_bias=use_bias
        )
        self.conv3d3_batch = layers.BatchNormalization(epsilon=1e-5)

        if shortcut == True:
            self.shortcut = ConvXD(
                filters * 4,
                1,
                (1, shortcut_stride, shortcut_stride),
                padding="same",
                use_bias=use_bias,
            )
        else:
            self.shortcut = None

    @tf.function
    def call(self, inputs, training=False):
        x = self.conv3d1(inputs)
        x = self.preact1(x, training)

        x = self.conv3d2(x)
        x = self.preact2(x, training)

        x = self.conv3d3(x)
        x = self.conv3d3_batch(x)

        if self.shortcut:
            x = layers.Add()([x, self.shortcut(inputs)])
        else:
            x = layers.Add()([x, inputs])

        x = layers.Activation(activation="swish")(x)

        return x


class SlowBody(tf.keras.layers.Layer):
    def __init__(self, stages, filters=64):
        super(SlowBody, self).__init__()
        self.stages = stages
        self.res_blocks = []
        self.concat = layers.Concatenate()

        self.init_block = InitBlock(filters, (1, 7, 7))
        for stage_num, conv_num in enumerate(stages):
            for conv in range(conv_num):
                time_kernel_size = 1 if stage_num < 1 else 3
                shortcut = True if conv == 0 else None
                if conv == 0:
                    if stage_num == 0:
                        self.res_blocks.append(
                            ResBlock(
                                filters,
                                time_kernel_size=time_kernel_size,
                                shortcut=shortcut,
                                stride=1,
                                shortcut_stride=1,
                            )
                        )
                    else:
                        self.res_blocks.append(
                            ResBlock(
                                filters,
                                time_kernel_size=time_kernel_size,
                                shortcut=shortcut,
                                stride=2,
                                shortcut_stride=2,
                            )
                        )
                else:
                    self.res_blocks.append(ResBlock(filters, shortcut=shortcut))
            filters = filters * 2
        self.global_avgpool3d = layers.GlobalAveragePooling3D()

    @tf.function
    def call(self, x, laterals, training=False):
        num_res_block = 0

        x = self.init_block(x)
        for conv_num, lateral in zip(self.stages, laterals):
            x = self.concat([x, lateral])
            for _ in range(conv_num):
                x = self.res_blocks[num_res_block](x, training)
                num_res_block += 1
        x = self.global_avgpool3d(x)

        return x


class FastBody(tf.keras.layers.Layer):
    def __init__(self, stages, filters: int = 8):
        super(FastBody, self).__init__()
        self.stages = stages
        self.filters = filters

        self.main_res_blocks = []
        self.last_res_blocks = []
        self.conv3s = []
        self.first_lateral = ConvXD(
            filters * 2,
            kernel_size=(5, 1, 1),
            strides=(8, 1, 1),
            padding="same",
            use_bias=False,
        )
        self.init_block = InitBlock(filters, (5, 7, 7))
        for stage_num, conv_num in enumerate(stages):
            for conv in range(conv_num):
                shortcut = True if conv == 0 else None
                if conv == 0:
                    if stage_num == 0:
                        filters = filters * 1
                        self.main_res_blocks.append(
                            ResBlock(
                                filters,
                                time_kernel_size=3,
                                shortcut=shortcut,
                                shortcut_stride=1,
                            )
                        )
                    else:
                        filters = filters * 2
                        self.main_res_blocks.append(
                            ResBlock(
                                filters,
                                time_kernel_size=3,
                                shortcut=shortcut,
                                stride=2,
                                shortcut_stride=2,
                            )
                        )
                else:
                    if stage_num == 0:
                        self.main_res_blocks.append(
                            ResBlock(filters, time_kernel_size=3, shortcut=None)
                        )
                    else:
                        self.main_res_blocks.append(
                            ResBlock(
                                filters,
                                time_kernel_size=3,
                                shortcut=None,
                                shortcut_stride=2,
                            )
                        )

            self.last_res_blocks.append(
                ResBlock(filters, time_kernel_size=3, shortcut=None)
            )
            self.conv3s.append(
                ConvXD(
                    filters * 8,
                    kernel_size=(5, 1, 1),
                    strides=(8, 1, 1),
                    padding="same",
                    use_bias=False,
                )
            )

        self.global_avg_pool3d = layers.GlobalAveragePooling3D()

    @tf.function
    def call(self, x, training=False):
        with tf.init_scope():
            laterals = []

        cnt_main_res_block = 0
        cnt_last_res_block = 0
        cnt_conv3s = 0
        x = self.init_block(x)

        first_lateral = self.first_lateral(x)
        laterals.append(first_lateral)

        for conv_num in self.stages:
            for _ in range(conv_num):
                x = self.main_res_blocks[cnt_main_res_block](x, training)
                cnt_main_res_block += 1

            x = self.last_res_blocks[cnt_last_res_block](x, training)
            cnt_last_res_block += 1

            lateral = self.conv3s[cnt_conv3s](x)
            cnt_conv3s += 1

            laterals.append(lateral)

        x = self.global_avg_pool3d(x)

        return x, laterals


class SlowFastNet(tf.keras.Model):
    resnet = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }

    def __init__(self, input_shapes, resnet_size=50, outputs=3):
        super(SlowFastNet, self).__init__()
        self.input_shapes = input_shapes
        self.fast_data_layer = DataLayer(2)
        self.slow_data_layer = DataLayer(16)

        self.fast_body = FastBody(
            self.resnet[resnet_size], filters=input_shapes[0] // 8
        )
        self.slow_body = SlowBody(self.resnet[resnet_size], filters=input_shapes[0])

        self.concat = layers.Concatenate()
        self.dropout = layers.Dropout(0.6)
        self.dense = layers.Dense(outputs, activation="softmax")

    @tf.function
    def call(self, x, training=False, mask=None):
        fast_inputs = self.fast_data_layer(x)
        slow_inputs = self.slow_data_layer(x)
        fast_x, laterals = self.fast_body(fast_inputs, training)
        slow_x = self.slow_body(slow_inputs, laterals, training)

        x = self.concat([slow_x, fast_x])
        x = self.dropout(x)
        x = self.dense(x)

        return x

    def build_graph(self):
        x = layers.Input(self.input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


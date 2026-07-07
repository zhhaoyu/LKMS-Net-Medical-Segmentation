import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, MaxPooling2D, UpSampling2D,
    BatchNormalization, Activation, Add, Concatenate, Multiply,
    Subtract, GlobalAveragePooling2D, Dense, Reshape, LeakyReLU
)
from tensorflow.keras.optimizers import Adam

from src.layers import SqueezeAndExcitation
from src.losses import combined_loss
from src.metrics import dice_coefficient, iou, f1_score, precision, recall


def context_encoding_module(input_tensor):
    conv1x1 = Conv2D(64, (1, 1), padding='same')(input_tensor)
    conv1x1 = BatchNormalization()(conv1x1)
    conv1x1 = Activation('relu')(conv1x1)

    conv3x3_1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(conv1x1)
    conv3x3_1 = BatchNormalization()(conv3x3_1)
    conv3x3_1 = Activation('relu')(conv3x3_1)
    conv3x3_1 = Concatenate(axis=-1)([conv3x3_1, conv1x1])

    conv3x3_2 = DepthwiseConv2D(
        kernel_size=(3, 3), padding='same', dilation_rate=(2, 2)
    )(conv3x3_1)
    conv3x3_2 = BatchNormalization()(conv3x3_2)
    conv3x3_2 = Activation('relu')(conv3x3_2)

    conv3x3_3 = DepthwiseConv2D(
        kernel_size=(3, 3), padding='same', dilation_rate=(3, 3)
    )(conv3x3_1)
    conv3x3_3 = BatchNormalization()(conv3x3_3)
    conv3x3_3 = Activation('relu')(conv3x3_3)
    conv3x3_3 = Add()([conv3x3_1, conv3x3_2, conv3x3_3])

    conv3x3_4 = Conv2D(64, (1, 1), padding='same')(conv3x3_3)
    conv3x3_4 = BatchNormalization()(conv3x3_4)
    conv3x3_4 = Activation('relu')(conv3x3_4)

    context_encoded = Add()([conv1x1, conv3x3_4])
    context_encoded = SqueezeAndExcitation(input_channels=64)(context_encoded)

    return context_encoded


def intermediate_module(input_tensor):
    conv1x1 = Conv2D(128, (1, 1), padding='same')(input_tensor)
    conv1x1 = BatchNormalization()(conv1x1)
    conv1x1 = Activation('relu')(conv1x1)

    conv5x5_1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(conv1x1)
    conv5x5_1 = BatchNormalization()(conv5x5_1)
    conv5x5_1 = Activation('relu')(conv5x5_1)
    conv5x5_1 = Concatenate(axis=-1)([conv5x5_1, conv1x1])

    conv5x5_2 = DepthwiseConv2D(
        kernel_size=(3, 3), padding='same', dilation_rate=(2, 2)
    )(conv5x5_1)
    conv5x5_2 = BatchNormalization()(conv5x5_2)
    conv5x5_2 = Activation('relu')(conv5x5_2)
    conv5x5_2 = Add()([conv5x5_1, conv5x5_2])

    conv5x5_3 = Conv2D(128, (1, 1), padding='same')(conv5x5_2)
    conv5x5_3 = BatchNormalization()(conv5x5_3)
    conv5x5_3 = Activation('relu')(conv5x5_3)

    intermediate_encoded = Add()([conv1x1, conv5x5_3])
    intermediate_encoded = SqueezeAndExcitation(input_channels=128)(intermediate_encoded)

    return intermediate_encoded


def local_encoding_module(input_tensor):
    conv1x1 = Conv2D(256, (1, 1), padding='same')(input_tensor)
    conv1x1 = BatchNormalization()(conv1x1)
    conv1x1 = Activation('relu')(conv1x1)

    conv7x7_1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(conv1x1)
    conv7x7_1 = BatchNormalization()(conv7x7_1)
    conv7x7_1 = Activation('relu')(conv7x7_1)
    conv7x7_1 = Concatenate(axis=-1)([conv7x7_1, conv1x1])

    conv7x7_2 = Conv2D(256, (1, 1), padding='same')(conv7x7_1)
    conv7x7_2 = BatchNormalization()(conv7x7_2)
    conv7x7_2 = Activation('relu')(conv7x7_2)

    local_encoded = Add()([conv1x1, conv7x7_2])
    local_encoded = SqueezeAndExcitation(input_channels=256)(local_encoded)

    return local_encoded


def guided_block(input_tensor1, input_tensor2):
    x1 = Conv2D(128, (1, 1), activation='relu')(input_tensor1)
    x2 = Conv2D(128, (1, 1), activation='relu')(input_tensor2)

    pooled = GlobalAveragePooling2D()(Add()([x1, x2]))
    weights = Dense(2, activation='softmax')(pooled)
    w1, w2 = tf.split(weights, num_or_size_splits=2, axis=-1)

    w1 = Reshape((1, 1, 1))(w1)
    w2 = Reshape((1, 1, 1))(w2)
    x1_weighted = Multiply()([x1, w1])
    x2_weighted = Multiply()([x2, w2])

    diff = Conv2D(128, (1, 1), activation='relu')(Subtract()([x1_weighted, x2_weighted]))

    combined = Add()([x1_weighted, x2_weighted])
    attn = GlobalAveragePooling2D()(combined)
    attn = Reshape((1, 1, attn.shape[-1]))(attn)
    attn = Conv2D(128, (1, 1), activation='sigmoid')(attn)

    rel = Multiply()([combined, attn])
    rel = LeakyReLU()(rel)

    output = Add()([diff, rel])

    return output


def fusion(path1, path2, path3, path4):
    fused_features = Concatenate(axis=-1)([path1, path2, path3, path4])

    conv3_1 = DepthwiseConv2D(
        kernel_size=(3, 1), padding='same', dilation_rate=(2, 2)
    )(fused_features)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    conv3_1 = Conv2D(32, (1, 1), padding='same')(conv3_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)

    conv3_2 = DepthwiseConv2D(
        kernel_size=(1, 3), padding='same', dilation_rate=(2, 2)
    )(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    conv3_2 = Conv2D(32, (1, 1), padding='same')(conv3_2)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    conv3_3 = DepthwiseConv2D(
        kernel_size=(3, 1), padding='same', dilation_rate=(3, 3)
    )(fused_features)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    conv3_3 = Conv2D(32, (1, 1), padding='same')(conv3_3)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)

    conv3_4 = DepthwiseConv2D(
        kernel_size=(1, 3), padding='same', dilation_rate=(3, 3)
    )(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    conv3_4 = Activation('relu')(conv3_4)
    conv3_4 = Conv2D(32, (1, 1), padding='same')(conv3_4)
    conv3_4 = BatchNormalization()(conv3_4)
    conv3_4 = Activation('relu')(conv3_4)

    conv3_5 = DepthwiseConv2D(
        kernel_size=(3, 1), padding='same'
    )(fused_features)
    conv3_5 = BatchNormalization()(conv3_5)
    conv3_5 = Activation('relu')(conv3_5)
    conv3_5 = Conv2D(32, (1, 1), padding='same')(conv3_5)
    conv3_5 = BatchNormalization()(conv3_5)
    conv3_5 = Activation('relu')(conv3_5)

    conv3_6 = DepthwiseConv2D(
        kernel_size=(1, 3), padding='same'
    )(conv3_5)
    conv3_6 = BatchNormalization()(conv3_6)
    conv3_6 = Activation('relu')(conv3_6)
    conv3_6 = Conv2D(32, (1, 1), padding='same')(conv3_6)
    conv3_6 = BatchNormalization()(conv3_6)
    conv3_6 = Activation('relu')(conv3_6)

    conv3_7 = Conv2D(32, (1, 1), padding='same')(fused_features)
    conv3_7 = BatchNormalization()(conv3_7)
    conv3_7 = Activation('relu')(conv3_7)

    fused_features = Concatenate(axis=-1)([conv3_2, conv3_4, conv3_6, conv3_7])

    return fused_features


def MF2_Net(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    path1 = context_encoding_module(x)
    path1_pooling = MaxPooling2D(pool_size=(2, 2))(path1)

    path2_im = intermediate_module(path1_pooling)
    path2_pooling = MaxPooling2D(pool_size=(2, 2))(path2_im)

    path3_lem = local_encoding_module(path2_pooling)
    path3_pooling = MaxPooling2D(pool_size=(2, 2))(path3_lem)

    gf1_3 = Activation('relu')(
        BatchNormalization()(Conv2D(16, (7, 7), padding='same')(x))
    )
    gf1_4 = Activation('relu')(
        BatchNormalization()(Conv2D(16, (9, 9), padding='same')(gf1_3))
    )
    gf1_5 = Add()([gf1_3, gf1_4])
    gf1_5 = Activation('relu')(
        BatchNormalization()(Conv2D(32, (7, 7), padding='same')(gf1_5))
    )
    gf1_6 = Add()([x, gf1_5])
    gf1_pooling = MaxPooling2D(pool_size=(2, 2))(gf1_6)

    gf2 = Activation('relu')(
        BatchNormalization()(Conv2D(64, (1, 1), padding='same')(gf1_pooling))
    )
    gf2_1 = Activation('relu')(
        BatchNormalization()(Conv2D(32, (3, 3), padding='same')(gf2))
    )
    gf2_2 = Activation('relu')(
        BatchNormalization()(Conv2D(16, (7, 7), padding='same')(gf2_1))
    )
    gf2_3 = Activation('relu')(
        BatchNormalization()(Conv2D(16, (11, 11), padding='same')(gf2_2))
    )
    gf2_4 = Add()([gf2_2, gf2_3])
    gf2_4 = Activation('relu')(
        BatchNormalization()(Conv2D(32, (7, 7), padding='same')(gf2_4))
    )
    gf2_5 = Add()([gf2_1, gf2_4])
    gf2_5 = Activation('relu')(
        BatchNormalization()(Conv2D(64, (3, 3), padding='same')(gf2_5))
    )
    gf2_6 = Add()([gf2, gf2_5])
    gf2_pooling = MaxPooling2D(pool_size=(2, 2))(gf2_6)

    gf3 = Activation('relu')(
        BatchNormalization()(Conv2D(128, (1, 1), padding='same')(gf2_pooling))
    )
    gf3_1 = Activation('relu')(
        BatchNormalization()(Conv2D(64, (3, 3), padding='same')(gf3))
    )
    gf3_2 = Activation('relu')(
        BatchNormalization()(Conv2D(32, (3, 3), padding='same')(gf3_1))
    )
    gf3_3 = Activation('relu')(
        BatchNormalization()(Conv2D(16, (7, 7), padding='same')(gf3_2))
    )
    gf3_4 = Activation('relu')(
        BatchNormalization()(Conv2D(16, (13, 13), padding='same')(gf3_3))
    )
    gf3_5 = Add()([gf3_3, gf3_4])
    gf3_5 = Activation('relu')(
        BatchNormalization()(Conv2D(32, (7, 7), padding='same')(gf3_5))
    )
    gf3_6 = Add()([gf3_2, gf3_5])
    gf3_6 = Activation('relu')(
        BatchNormalization()(Conv2D(64, (3, 3), padding='same')(gf3_6))
    )
    gf3_7 = Add()([gf3_1, gf3_6])
    gf3_7 = Activation('relu')(
        BatchNormalization()(Conv2D(128, (3, 3), padding='same')(gf3_7))
    )
    gf3_8 = Add()([gf3, gf3_7])
    gf3_pooling = MaxPooling2D(pool_size=(2, 2))(gf3_8)

    path1_pooling_adjusted = Conv2D(256, (1, 1), padding='same')(path1_pooling)
    path1_pooling_adjusted = BatchNormalization()(path1_pooling_adjusted)
    path1_pooling_adjusted = Activation('relu')(path1_pooling_adjusted)
    path1_pooling_adjusted_pooling = MaxPooling2D(pool_size=(4, 4))(path1_pooling_adjusted)

    path2_pooling_adjusted = Conv2D(256, (1, 1), padding='same')(path2_pooling)
    path2_pooling_adjusted = BatchNormalization()(path2_pooling_adjusted)
    path2_pooling_adjusted = Activation('relu')(path2_pooling_adjusted)
    path2_pooling_adjusted_pooling = MaxPooling2D(pool_size=(2, 2))(path2_pooling_adjusted)

    gf3_pooling = Conv2D(256, (1, 1), padding='same')(gf3_pooling)
    gf3_pooling = BatchNormalization()(gf3_pooling)
    gf3_pooling = Activation('relu')(gf3_pooling)

    fused_features = fusion(
        path1_pooling_adjusted_pooling,
        path2_pooling_adjusted_pooling,
        path3_pooling,
        gf3_pooling
    )

    up1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(fused_features)
    guided_out1 = guided_block(path3_lem, gf3_8)

    decoded_path1 = Concatenate(axis=-1)([guided_out1, up1])
    decoded_path1 = Conv2D(128, (3, 3), padding='same')(decoded_path1)
    decoded_path1 = BatchNormalization()(decoded_path1)
    decoded_path1 = Activation('relu')(decoded_path1)

    up2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(decoded_path1)
    guided_out2 = guided_block(path2_im, gf2_6)
    guided_out2 = Conv2D(64, (1, 1), activation='relu', padding='same')(guided_out2)

    decoded_path2 = Concatenate(axis=-1)([guided_out2, up2])
    decoded_path2 = Conv2D(64, (3, 3), padding='same')(decoded_path2)
    decoded_path2 = BatchNormalization()(decoded_path2)
    decoded_path2 = Activation('relu')(decoded_path2)

    up3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(decoded_path2)
    guided_out3 = guided_block(path1, gf1_6)
    guided_out3 = Conv2D(32, (1, 1), activation='relu', padding='same')(guided_out3)

    decoded_path3 = Concatenate(axis=-1)([guided_out3, up3])
    decoded_path3 = Conv2D(32, (3, 3), padding='same')(decoded_path3)
    decoded_path3 = BatchNormalization()(decoded_path3)
    decoded_path3 = Activation('relu')(decoded_path3)

    output = Conv2D(1, (1, 1), activation='sigmoid')(decoded_path3)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=['accuracy', f1_score, iou, dice_coefficient, precision, recall]
    )

    return model

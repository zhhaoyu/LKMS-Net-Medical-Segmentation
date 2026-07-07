import tensorflow as tf


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def bce_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + 0.5 * bce_loss(y_true, y_pred)

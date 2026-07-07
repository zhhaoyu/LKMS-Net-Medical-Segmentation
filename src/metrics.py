import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def precision(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positive = tf.reduce_sum(y_true * y_pred)
    predicted_positive = tf.reduce_sum(y_pred)
    return (true_positive + smooth) / (predicted_positive + smooth)


def recall(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positive = tf.reduce_sum(y_true * y_pred)
    actual_positive = tf.reduce_sum(y_true)
    return (true_positive + smooth) / (actual_positive + smooth)


def f1_score(y_true, y_pred, smooth=1e-6):
    p = precision(y_true, y_pred, smooth)
    r = recall(y_true, y_pred, smooth)
    return 2.0 * (p * r) / (p + r + smooth)

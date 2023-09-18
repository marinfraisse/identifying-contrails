
import tensorflow as tf


def proba_to_pixel(y):
    return tf.where(y > 0.5, tf.ones_like(y),tf.zeros_like(y))
def dice_metric(y_true, y_pred):
    y_pred = proba_to_pixel(y_pred)
    y_true = proba_to_pixel(y_true)
    smooth = 1e-5
    y_true_sum = tf.reduce_sum(y_true)
    y_pred_sum = tf.reduce_sum(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = y_true_sum + y_pred_sum
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_sum = tf.reduce_sum(y_true)
    y_pred_sum = tf.reduce_sum(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = y_true_sum + y_pred_sum
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice
def binary_crossentropy(y_true, y_pred) :
    return (-1)*tf.math.reduce_sum(y_true * tf.math.log(y_pred + 1e-7)  + (1-y_true)*tf.math.log(1-y_pred + 1e-7))
def weighted_binary_crossentropy_func(weight_of_1) :
    def weighted_binary_crossentropy(y_true, y_pred):
        return (-1)*tf.math.reduce_sum((1-weight_of_1)*y_true * tf.math.log(y_pred + 1e-7)  + weight_of_1*(1-y_true)*tf.math.log(1-y_pred + 1e-7))
    return weighted_binary_crossentropy

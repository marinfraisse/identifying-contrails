# imports
import os
import random
import datetime
import csv

# Data analysis and manipulation
import numpy as np

# # Data visualization
# from matplotlib import animation
# import matplotlib.pyplot as plt
# from IPython import display
# import seaborn as sns
# import plotly

# ML, DL & Modelling
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import callbacks
# from tensorflow.keras.models import Sequential

# Garbage collect for generator
# import gc
import random

#import parameters
from idcontrails.params import *

from idcontrails.ml_logic.metrics import dice_metric, dice_loss, binary_crossentropy


def build_unet_model(input_layer, start_neurons, drop_out_factor):
    # Downsampling path / Decoder
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(0.25 * drop_out_factor)(pool1)

    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(0.5 * drop_out_factor)(pool2)

    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(0.5 * drop_out_factor)(pool3)

    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(0.5 * drop_out_factor)(pool4)

    # Middle path / Bottleneck
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # Upsampling / Decoder using Transpose convolution
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Dropout(0.5 * drop_out_factor)(uconv4)
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    uconv3 = tf.keras.layers.Dropout(0.5 * drop_out_factor)(uconv3)
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
    uconv2 = tf.keras.layers.Dropout(0.5 * drop_out_factor)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    uconv1 = tf.keras.layers.Dropout(0.5 * drop_out_factor)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer

def load_model() :
    print('-')
    print('-')
    print('-')
    print('reloading model with latest manually added checkpoint')
    input_layer = tf.keras.layers.Input((IMG_SIZE_TARGET, IMG_SIZE_TARGET, NUMBER_CHANNELS_TARGET))
    output_layer = build_unet_model(input_layer, START_NEURONS, DROPOUT_RATIO)

    # U-Net model with Functional API from Keras
    model = tf.keras.Model(input_layer, output_layer, name=MODEL_NAME)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=MAX_LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

    model.compile(optimizer=optimizer,
                    loss=dice_loss,  # Use the specified loss function
                    metrics=[dice_metric, dice_loss, binary_crossentropy])  # Add appropriate metrics

    model.load_weights(TF_CHECKPOINT_PATH).expect_partial()
    print("weights successfully loaded !" )
    print('-')
    print('-')
    print('-')
    return model

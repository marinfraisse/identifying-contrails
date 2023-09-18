# imports
import os
import random
import datetime
import csv

# Data analysis and manipulation
import numpy as np
import pandas as pd

# Data visualization
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython import display
import seaborn as sns
# import plotly

# ML, DL & Modelling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential

# Garbage collect for generator
import gc
import random

#import parameters
from idcontrails.params import *
from idcontrails.ml_logic.building_models import build_unet_model
from idcontrails.ml_logic.metrics import dice_metric, dice_loss, binary_crossentropy

from idcontrails.ml_logic.preprocessing import create_list_samples_with_contrails


# creating model architecture
if RELOAD_MODEL :
    print('-')
    print('-')
    print('-')
    print('reloading model with latest manually added checkpoint')
    input_layer = layers.Input((IMG_SIZE_TARGET, IMG_SIZE_TARGET, NUMBER_CHANNELS_TARGET))
    output_layer = build_unet_model(input_layer, START_NEURONS, DROPOUT_RATIO)

    # U-Net model with Functional API from Keras
    model = tf.keras.Model(input_layer, output_layer, name=MODEL_NAME)
    optimizer = tf.keras.optimizers.Adam(learning_rate=MAX_LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)



    # unet_model.compile(optimizer=optimizer,
    #                    loss=dice_loss,  # Use the specified loss function
    #                    metrics=[dice_metric, dice_loss, binary_crossentropy])  # Add appropriate metrics

    optimizer = tf.keras.optimizers.Adam(learning_rate=MAX_LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

    model.compile(optimizer=optimizer,
                    loss=dice_loss,  # Use the specified loss function
                    metrics=[dice_metric, dice_loss, binary_crossentropy])  # Add appropriate metrics

    model.load_weights(TF_CHECKPOINT_PATH).expect_partial()
    print("weights successfully loaded !" )
    print('-')
    print('-')
    print('-')

contrail_record_ids = create_list_samples_with_contrails()

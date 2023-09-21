# imports
import os
# import random
# import datetime
# import csv

# Data analysis and manipulation
import numpy as np
# import pandas as pd

# Data visualization
# from matplotlib import animation
import matplotlib.pyplot as plt
# from IPython import display
# import seaborn as sns
# import plotly

# ML, DL & Modelling
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers, callbacks
# from tensorflow.keras.models import Sequential

# Garbage collect for generator
# import gc
# import random
import requests
# import json
#import parameters
from idcontrails.params import *
# from idcontrails.ml_logic.building_models import load_model
from idcontrails.ml_logic.metrics import dice_metric, dice_loss, binary_crossentropy

from idcontrails.ml_logic.preprocessing import create_list_samples_with_contrails, loading_single_array
from idcontrails.ml_logic.plotting_contrails import load_random_image_and_mask, plot_results
from idcontrails.interface.api_call import api_call_predict


# creating model architecture
# if RELOAD_MODEL :
#     model = load_model()

# if TEST_PLOT :
#     for index in range(len(os.listdir(DATASET_SAMPLE_PATH)) ) :
#         input_image, output_mask, predicted_mask_image = load_random_image_and_mask(model, index)
#         nom = f'coucou{index}.png'

#         fig = plot_results(input_image, output_mask, predicted_mask_image )
#         # plt.savefig('coucou.png')
#         plt.savefig(os.path.join(FIG_SAVES_PATH, nom ))
#         print(nom + ' successfully saved')



# if TEST_API :
#     X = loading_single_array(index=1)
#     print(api_call_predict(X))
#     print(api_call_predict(X).shape)

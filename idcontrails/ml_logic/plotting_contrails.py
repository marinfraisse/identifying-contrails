# Choosing a random image from sampling set and defining the corresponding true mask and predicted mask
import tensorflow as tf
import numpy as np
import os
from idcontrails.params import *
import matplotlib.pyplot as plt


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def load_random_image_and_mask(model, index):

    dataset_sample_path = DATASET_SAMPLE_PATH

    # Choosing a random image from validation set

    record_list = os.listdir(dataset_sample_path)
    record_id = record_list[index]

    # loading 3 bands required for normalization and the mask
    with open(os.path.join(dataset_sample_path, record_id, 'band_11.npy'), 'rb') as f:
        band11 = np.load(f)
    with open(os.path.join(dataset_sample_path, record_id, 'band_14.npy'), 'rb') as f:
        band14 = np.load(f)
    with open(os.path.join(dataset_sample_path, record_id, 'band_15.npy'), 'rb') as f:
        band15 = np.load(f)
    with open(os.path.join(dataset_sample_path, record_id, 'human_pixel_masks.npy'), 'rb') as f:
        output_mask = np.load(f)

    # normalizing the selected image to plot it in RGB ash
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    image_full_sequences = np.clip(np.stack([r, g, b], axis=2), 0, 1)

    # choosing only the 5th sequence to plot the image
    N_TIMES_BEFORE = 4
    input_image = image_full_sequences[..., N_TIMES_BEFORE]

    # artificially adding 1 dimension to feed to the model
    input_image_model = tf.expand_dims(input_image, 0)

    # Use the model to predict a mask on the input image
    predicted_mask = model.predict(input_image_model)

    # Removing the additional dimension to plot the image
    predicted_mask_image = predicted_mask[0,:,:,:]

    return input_image, output_mask, predicted_mask_image

def plot_results(input_image, output_mask, predicted_mask_image):

    # plotting the results on a graph
    plt.figure(figsize=(18, 6))
    ax = plt.subplot(2, 3, 1)
    ax.imshow(input_image)
    ax.set_title('Input image')

    ax = plt.subplot(2, 3, 2)
    ax.imshow(output_mask, interpolation='none')
    ax.set_title('Contrail mask from validation set')

    ax = plt.subplot(2, 3, 3)
    ax.imshow(input_image)
    ax.imshow(output_mask, cmap='Reds', alpha=.4, interpolation='none')
    ax.set_title('Contrail mask on input image')

    ax = plt.subplot(2, 3, 4)
    ax.imshow(input_image)
    ax.set_title('Input image')

    ax = plt.subplot(2, 3, 5)
    ax.imshow(predicted_mask_image, interpolation='none')
    ax.set_title('Predicted contrail mask using model')

    ax = plt.subplot(2, 3, 6)
    ax.imshow(input_image)
    ax.imshow(predicted_mask_image, cmap='Reds', alpha=.4, interpolation='none')
    ax.set_title('Predicted contrail mask on input image')
    plt.show()
    fig = plt.gcf()
    return fig

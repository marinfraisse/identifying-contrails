import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

N_TIMES_BEFORE = 4

# Defining bounds for each band
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


# Defining X normalization function

def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def build_dataset():
    data_sample_path = os.path.join(os.path.dirname(__file__), "../" , "app_assets", "dataset_sample_app" )
    record_ids = os.listdir(data_sample_path)
    print(record_ids)

    X_pred_list = []
    y_true_images = []
    for record_id in record_ids:
        # loading 3 bands required for normalization and the mask
        with open(os.path.join(data_sample_path, record_id, 'band_11.npy'), 'rb') as f:
            band11 = np.load(f)
        with open(os.path.join(data_sample_path, record_id, 'band_14.npy'), 'rb') as f:
            band14 = np.load(f)
        with open(os.path.join(data_sample_path, record_id, 'band_15.npy'), 'rb') as f:
            band15 = np.load(f)
        with open(os.path.join(data_sample_path, record_id, 'human_pixel_masks.npy'), 'rb') as f:
            y_true = np.load(f)

        # Normalizing the selected image to plot it in RGB ash
        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)
        image_full_sequences = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        # Choosing only the 5th sequence to plot the image
        input_image = image_full_sequences[..., N_TIMES_BEFORE]

        # Artificially adding 1 dimension to feed to the model
        X_pred = tf.expand_dims(input_image, 0)

        # Adding the X_pred into the list
        X_pred_list.append(X_pred)

        # Adding y_true to the list
        y_true_images.append(y_true)

    return X_pred_list, y_true_images


def pred_debile(index):

    # Choosing X_pred and y_true
    X_pred_list, y_true_images = build_dataset()

    X_pred = X_pred_list[index]
    y_true_image = y_true_images[index]

    # Use the model to predict a mask on the input image
    y_pred = (-1) * y_true_image
    print(y_pred.shape)

    # Removing the additional dimension to plot the image
    X_pred_image = X_pred[0,:,:,:]
    #y_pred_image = y_pred[0,:,:,:]
    y_pred_image = y_pred

    return X_pred_image, y_pred_image, y_true_image







# # plotting the results on a graph

# plt.figure(figsize=(18, 6))
# ax = plt.subplot(2, 3, 1)
# ax.imshow(input_image)
# ax.set_title('Input image')

# ax = plt.subplot(2, 3, 2)
# ax.imshow(y_true, interpolation='none')
# ax.set_title('Contrail mask from validation set')

# ax = plt.subplot(2, 3, 3)
# ax.imshow(input_image)
# ax.imshow(y_true, cmap='Reds', alpha=.4, interpolation='none')
# ax.set_title('Contrail mask on input image')

# plt.figure(figsize=(18, 6))
# ax = plt.subplot(2, 3, 4)
# ax.imshow(input_image)
# ax.set_title('Input image')

# ax = plt.subplot(2, 3, 5)
# ax.imshow(y_pred_image, interpolation='none')
# ax.set_title('Predicted contrail mask using model')

# ax = plt.subplot(2, 3, 6)
# ax.imshow(input_image)
# ax.imshow(y_pred_image, cmap='Reds', alpha=.4, interpolation='none')
# ax.set_title('Predicted contrail mask on input image');

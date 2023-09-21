import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

N_TIMES_BEFORE = 4

# Defining bounds for each band
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


# Defining X normalization function

def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def build_dataset():
    data_sample_path = os.path.join(os.path.dirname(__file__), "app_assets", "dataset_sample_app")
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

def plot_results_streamlit(input_image, y_true_image, y_pred_image, final_layer_true, final_layer_pred):

    # Preparing the superpositioned masks

    # Building the layout with 3 columns
    col_1, col_2,col_3 = st.columns(3)

    # Displaying images in each column
    with col_1:
        fig_1 = px.imshow(input_image)
        fig_1.update_layout(width=300, height=300)
        st.text(f'Original image')
        st.plotly_chart(fig_1, width=300, height=300)
        st.markdown("***")
        fig_2 = px.imshow(input_image)
        fig_2.update_layout(width=300, height=300)
        st.text(f'Original image')
        st.plotly_chart(fig_1, width=300, height=300)

    with col_2:
        fig_3 = px.imshow(y_true_image)
        fig_3.update_layout(width=300, height=300)
        fig_3.update(layout_coloraxis_showscale=False)
        st.text(f'Original contrail mask')
        st.plotly_chart(fig_3, width=300, height=300)
        st.markdown("***")
        fig_4 = px.imshow(y_pred_image)
        fig_4.update_layout(width=300, height=300)
        fig_4.update(layout_coloraxis_showscale=False)
        st.text(f'Your prediction mask')
        st.plotly_chart(fig_4, width=300, height=300)


    with col_3:
        fig_5 = px.imshow(final_layer_true)
        fig_5.update_layout(width=300, height=300)
        # fig_5.update(layout_coloraxis_showscale=False)
        st.text(f'Final layer true')
        st.plotly_chart(fig_5, width=300, height=300)
        st.markdown("***")
        fig_6 = px.imshow(final_layer_pred)
        fig_6.update_layout(width=300, height=300)
        # fig_6.update(layout_coloraxis_showscale=False)
        st.text(f'Predicted final layer')
        st.plotly_chart(fig_6, width=300, height=300)

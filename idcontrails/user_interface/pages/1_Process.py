# Imports & variables
import os
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from app_processing import *

# Creating the absolute path for assets
absolute_path_root_old = '/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/app_assets'
absolute_path_root = os.path.join(os.path.dirname(__file__), "../","app_assets" )

# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Fraisse Tech - Identifying contrails",
    page_icon = ":strawberry:",
    initial_sidebar_state = 'auto')


########################################################## Contrail gallery ##########################################################

# Displaying top image with logo
st.image(os.path.join(absolute_path_root, 'fraisse_tech_background.jpg'), caption='Fraisse Techologies Incorporated®')

# Writing the site body
st.header('Dataset')

st.subheader('Presenting the dataset')

st.write("We used a dataset composed of 2 subsets of data")
st.markdown("""
            - Contrail images: original satellite images (256x256x8) on which we trained our model to identify contrails
            - Labeled contrails : binary mask derived from original satelite images based on 4+ different labelers annotating each image, 1 if a pixel displayed contrails and 0 if the image did not display contrails. This labeled image is called the 'Ground truth'
            """)
st.image(os.path.join(absolute_path_root,'false_color_and_mask.png'), caption='Normalized satellite image on the left, labeled contrail on the right')

st.subheader('Preprocessing')
st.write('We normalized the images in order to vizualize contrails properly, using an "ash" color scheme - originally developed for viewing volcanic ash in the atmosphere and also useful to view contrails which appear in dark blue (see image above)')

st.write('The normalization function is an idea we came up with during a heated brainstorming session mobilizing all of Fraisse Inc. staff for hours - Just kidding we used the methodology developed in the following [scientific paper](https://eumetrain.org/sites/default/files/2020-05/RGB_recipes.pdf) (p.7)')

st.markdown("***")

st.header('Our Model')

st.subheader('Training in chunks')
st.write("The main issue we faced was the large lenght of the dataset (450Go, seriously bro?) which made the model training too complex and time-consuming to handle for our poor little computers. The solution we found was to split the dataset in 10 chunks, enabling us to train the model on smaller subsets of data, delete them from our computer's memory before starting again the training on the next chunk.")
st.write("")
code = '''
# Defining the list of chunks
list_chuncks = []
chunck_names = []
for i in range(0, NB_CHUNCKS):
    chunck_name = f'chunck_{i}'
    chunk_i_records = contrail_record_ids[(i*CHUNCK_SIZE):((i+1)*CHUNCK_SIZE)]
    chunck_names.append(chunck_name)
    list_chuncks.append(chunk_i_records)
chuncks_record_dict = dict(zip(chunck_names, list_chuncks))
'''
with st.expander("Expand to see detailed code for splitting data in chunks"):
    st.code(code, language="python")

code_2 = '''
# Defining a function to load chunks
def load_normalize_X_chunck(chunck, BASE_DIR, band_choice, N_TIMES_BEFORE):
    X_chunck = []
    for record_id in chuncks_record_dict[chunck]:
        # Building band paths
        record_first_band_path = os.path.join(BASE_DIR, record_id, band_choice[0])
        record_second_band_path = os.path.join(BASE_DIR, record_id, band_choice[1])
        record_third_band_path = os.path.join(BASE_DIR, record_id, band_choice[2])
        # Loading each band
        first_band = np.load(open(record_first_band_path, 'rb'))[:, :, N_TIMES_BEFORE]
        second_band = np.load(open(record_second_band_path, 'rb'))[:, :, N_TIMES_BEFORE]
        third_band = np.load(open(record_third_band_path, 'rb'))[:, :, N_TIMES_BEFORE]
        # Normalizing each band with its relevant bounds
        _T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)
        # Applying normalization functions
        normalized_r = normalize_range(third_band - second_band, _TDIFF_BOUNDS)
        normalized_g = normalize_range(second_band - first_band, _CLOUD_TOP_TDIFF_BOUNDS)
        normalized_b = normalize_range(second_band, _T11_BOUNDS)
        # Building a single record from all bands
        record = np.clip(np.stack([normalized_r, normalized_g, normalized_b], axis=2), 0, 1)
        # Appending chunck list
        X_chunck.append(record)
    # Building the chunck array
    X_chunck_array = np.stack(X_chunck, axis=0)
    return X_chunck_array
'''
with st.expander("Expand to see detailed code for training with chunks"):
    st.code(code_2, language="python")

st.subheader('Defining a custom loss and performance metric')
st.write("For this project we had to optimize the Dice metric, which measures the proportion of well predicted contrails within the set of total contrails")
st.latex(r'\text{Dice}(X, Y) = \frac{2 \cdot |X \cap Y|}{|X| + |Y|}')
st.write("")
st.write("We also built a custom loss function which we called the Dice loss, based on the same formula as the Dice metric")
code_3 = '''
# Defining the dice metric
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
'''

code_5 = '''
# Defining the dice loss
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_sum = tf.reduce_sum(y_true)
    y_pred_sum = tf.reduce_sum(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = y_true_sum + y_pred_sum
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice
'''
with st.expander("Expand to see detailed code for dice metric"):
    st.code(code_3, language="python")
with st.expander("Expand to see detailed code for dice loss"):
    st.code(code_5, language="python")

st.subheader('Model architecture')
st.write("Finally we chose a U-Net model which is very efficient for image segmentation task. This model's architecture is composed of 4 layers")
st.markdown("""
            1. Encoder: mulitple layers of CNN responsible for learning features in the input image. these layers reduce the image dimension (maxpooling) while increasing the number of channels
            2. Bottleneck: last CNN layer without maxpooling to keep the same image size and extract the most detailed features from the image
            3. Decoder: counterpart to the encoder - uses transposed CNN to expand the image size. This is combined with a concatenation step to bring back information learned with the encoder and make sure we don't lose info while expanding the image
            4. Output layer: single CNN with a sigmoid activation function to output a binary mask
            """)
st.image(os.path.join(absolute_path_root,'Unet_model.webp'))
code_4 = '''
# Defining the U-Net model
def build_model(input_layer, start_neurons, drop_out_factor):
    # Downsampling path / Decoder
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.25 * drop_out_factor)(pool1)

    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5 * drop_out_factor)(pool2)

    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5 * drop_out_factor)(pool3)

    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5 * drop_out_factor)(pool4)

    # Middle path / Bottleneck
    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # Upsampling / Decoder using Transpose convolution
    deconv4 = layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5 * drop_out_factor)(uconv4)
    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5 * drop_out_factor)(uconv3)
    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5 * drop_out_factor)(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5 * drop_out_factor)(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer
'''

with st.expander("Expand to see detailed code for U-Net model"):
    st.code(code_4, language="python")

st.subheader('Performance and limitations')
st.write("We achieve a performance of 0.57 for the Dice metric with our U-Net model, meaning that approximately 60% of all contrails were predicted correctly. We could explore additional methods to increase our model performance notably combining our dice loss with a weighted binary-crossentropy loss, increasing the number of channels fed to the model (e.g. 10 vs 3 for our current model) or exploring other models like a MaskRCNN")

# Ending line
st.markdown("***")
st.write("🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪")

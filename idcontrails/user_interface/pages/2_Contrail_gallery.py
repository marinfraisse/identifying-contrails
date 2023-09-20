# Imports & variables
import os
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from idcontrails.user_interface.app_processing import *

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

st.title("Contrail Gallery")
st.write("")
st.write("Our U-Net model was trained using satellite images, with a superposition of layers with various bandwiths allowing to capture different types of information.")
st.write("")
st.write("Here is a sample of what the training material looks like:")



# Creating the image dataset from local images
X_pred_list, y_true_images = build_dataset()

# Creating a figure and subplot space for the gallery
col_a, col_b = st.columns(2)

with col_a:
    for i in range(0, 20, 2): # Keeping only odd numbers
        st.markdown('***')
        img = X_pred_list[i][0,:,:,:] # Reshaping the images from the model format to the imshow format
        fig = px.imshow(img)
        fig.update_layout(width=400, height=400)
        st.text(f'Contrail example {i+1}')
        st.plotly_chart(fig, width=400, height=400)


with col_b:
    for i in range(1, 20, 2): # Keeping only even numbers
        st.markdown('***')
        img = X_pred_list[i][0,:,:,:] # Reshaping the images from the model format to the imshow format
        fig = px.imshow(img)
        fig.update_layout(width=400, height=400)
        st.text(f'Contrail example {i+1}')
        st.plotly_chart(fig, width=400, height=400)


st.markdown('***')
st.write("📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹")

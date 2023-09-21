# Imports & variables
import os
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from app_processing import *

# Creating the absolute path for assets
absolute_path_root = os.path.join(os.path.dirname(__file__), "app_assets" )
#absolute_path_root = os.path.join(os.path.dirname(__file__), "../","app_assets" )

# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Fraisse Tech - Identifying contrails",
    page_icon = ":strawberry:",
    initial_sidebar_state = 'auto')


########################################################## Team and value page ##########################################################

# Displaying top image with logo
st.image(os.path.join(absolute_path_root, 'fraisse_tech_background.jpg'), caption='Fraisse Techologies Incorporated®')

st.header("Let's show one last thin: our most successful model")
st.subheader("We got amazing results from a trained LSTM...")
st.subheader("(but it got very costly 💸💳)")

st.markdown("***")


if 'troll_choice' not in st.session_state:
    st.session_state['troll_choice'] = "Nothing"

st.subheader("Which output do you want to see?")
choice_of_troll = st.radio("Choose your version", ("Select", "Version 1", "Version 2", "Version 3"))
st.session_state['troll_choice'] = choice_of_troll

if st.session_state['troll_choice'] == "Version 1":
    troll_image = os.path.join(absolute_path_root, 'test_chops_troll_1.png')
    # fig = px.imshow(troll_image)
    # st.plotly_chart(fig, use_container_width=False)
    st.image(troll_image, caption="ChopAI")

elif st.session_state['troll_choice'] == "Version 2":
    troll_image = os.path.join(absolute_path_root, 'test_chops_troll_2.png')
    # fig = px.imshow(troll_image)
    # st.plotly_chart(fig, use_container_width=True)
    st.image(troll_image, caption="ChopAI")

elif st.session_state['troll_choice'] == "Version 3":
    troll_image = os.path.join(absolute_path_root, 'test_chops_troll_3.jpg')
    # fig = px.imshow(troll_image)
    # st.plotly_chart(fig, use_container_width=True)
    st.image(troll_image, caption="ContrAI")



st.markdown("***")

st.write("🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹🎵🎹")

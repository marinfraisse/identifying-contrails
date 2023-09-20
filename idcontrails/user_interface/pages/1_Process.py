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





st.markdown("***")
st.write("🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪🥼🧪")

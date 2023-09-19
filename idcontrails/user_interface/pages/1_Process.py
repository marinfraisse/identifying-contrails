# Imports & variables
import os
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from idcontrails.user_interface.temporary_plot import *

# Creating the absolute path for assets
absolute_path_root = '/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/app_assets'

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

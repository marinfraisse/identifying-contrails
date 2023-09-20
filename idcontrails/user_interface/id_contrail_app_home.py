# Imports & variables
import os
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from idcontrails.user_interface.app_processing import *
from st_pages import Page, show_pages

# Creating the absolute path for assets
absolute_path_root = '/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/app_assets'

# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Fraisse Tech - Identifying contrails",
    page_icon = ":strawberry:",
    initial_sidebar_state = 'auto')


# Changing pages names

show_pages(
    [
        Page("/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/id_contrail_app_home.py", "Context & objectives", "🎯"),
        Page("/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/pages/1_Process.py", "Our methodology and process", "🥼"),
        Page("/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/pages/2_Contrail_gallery.py", "Contrail Gallery", "📷"),
        Page("/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/pages/3_Model_demo.py", "Model demo - Spot the difference", "🔍"),
        Page("/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/pages/4_About_us.py", "About us", "🚀"),
    ]
)

# Objective is to display several pages
#  1 - Context and explanations ✅
#  2 - Our process and how we built the app
#  3 - Model demo and Spot the Difference
#  4 - Contrail gallery ✅
#  5 - About us (incl. team) ✅


########################################################## Context page ##########################################################

# Displaying top image with logo
st.image(os.path.join(absolute_path_root, 'fraisse_tech_background.jpg'), caption='Fraisse Techologies Incorporated®')


st.title("Identifying contrails to reduce global warming")
st.write("")

st.header("First up: what are contrails?")
st.write("Contrails are clouds of ice crystals that form in aircraft engine exhaust. Contrails - short for ‘condensation trails’ - are line-shaped clouds of ice crystals that form in aircraft engine exhaust, and are created by airplanes flying through super humid regions in the atmosphere.")
st.write("")


col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.image(os.path.join(absolute_path_root, 'contrails_left_to_right.jpg'), caption="Plane fart")

with col_2:
    st.image(os.path.join(absolute_path_root, 'sunset_contrails.jpg'), caption="A beautiful danger")

with col_3:
    st.image(os.path.join(absolute_path_root, 'contrails_top_gun.jpg'), caption="Tom Cruise making contrails")

st.markdown("***")

st.header("What does it have to do with global warming?")
st.write("Contrails contribute to global warming by trapping heat in the atmosphere. They block heat that normally is released from the earth overnight.  You can consider them as temporary greenhouse clouds that retain heat in the atmosphere. The issue is that with constant air traffic, this temporary effect becomes almost permanent, with huge environmental impact!")
st.write("Contrary to general belief, emissions from burning kerosene do not constitute the bulk of aviation's emissions. Persistent contrails contribute as much to global warming as the fuel that is burnt for flights! After years of research it is now well accepted that contrails contribute approximately 1% of all human caused global warming.")
st.write("")
st.image(os.path.join(absolute_path_root, 'contrail_stats.png'), caption="Source: Google Research, BCG Analysis 😂🎯")
st.markdown("***")

st.header("Keeping both feet on the ground: how can we leverage DL to mitigate the impact from contrails?")
st.write("Contrail avoidance is potentially one of the most scalable, cost-effective sustainability solutions available to airlines today.")
st.write("Researchers have developed models to predict when contrails will form and how much warming they will cause. However, they need to validate these models with satellite imagery. With reliable verification, pilots can have confidence in the models and the airline industry can have a trusted way to measure successful contrail avoidance.")
st.write("The endgame is to have models sufficiently reliable to empower and incetivize airlines to adapt their routes based on each flights risk to generate contrails, based on weather conditions.")
st.markdown("***")

st.write("✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈")

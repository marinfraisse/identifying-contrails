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


########################################################## Team and value page ##########################################################

# Displaying top image with logo
st.image(os.path.join(absolute_path_root, 'fraisse_tech_background.jpg'), caption='Fraisse Techologies Incorporated®')

st.title("Our mission")
st.subheader("Bringing peace-of-mind to careless frequent flyers")
st.markdown("***")
st.write("Air travel is not set to diminish as a whole. Even if travelers individually decide to change their habits and reduce the number of flights they board, there are more and more air travelers each year.")
st.write("Tech-driven initiatives to clean air travel - SAF or hydrogen-powered planes -  are still nascent and will take years to be developed and scaled, with airlines having little to no direct impact.")
st.write("Our mission is to help accelerate air travel decarbonization by having a pragmatic and simple way to tackle contrail generation through route management, responsibilizing air carriers in the process.")
st.markdown("***")

st.title("Our commitment")
st.subheader("Never stop working until we raise enough to exit")
st.markdown("***")
st.write("Bringing together the best of our knowledge and battle-tested expertise, we want to build a robust product, reliable and scalable through a continuous improvement mindset.")
st.write("Hard work and rigor are at the core of our DNA, and we are committed to bringint to life a great product through carefully crafted strategy and solid delivery.")
st.markdown("***")


st.title("Our team")
st.subheader("Supercharging Green Transition with God-given Talent")
st.markdown("***")


column_1, column_2, column_3 = st.columns(3)

with column_1:
    st.image(os.path.join(absolute_path_root,'marin_round.png'), width=200)
    st.header("Marin Fraisse")
    st.subheader("Founder & CEO")
    st.markdown("***")
    st.write("Marin started his carreer as a strategy consultant at BCG in Paris because he was craving for impact and wanted to apply the rigorous and robust methods he acquired in his scientific studies. #Striving4Greatness #Science")


with column_2:
    st.image(os.path.join(absolute_path_root,'robin_round.png'), width=200)
    st.header("Robin Diligent")
    st.subheader("Founder & COO")
    st.markdown("***")
    st.write("Robin started his carreer as a strategy consultant at BCG. He has been appointed COO because he has very strong arms (according to a former colleague) and will thus carry the team. #KOstaud #NoPainNoGains")

with column_3:
    st.image(os.path.join(absolute_path_root,'nico_round.png'), width=200)
    st.header("Nico Berretti")
    st.subheader("Founder & Intern")
    st.markdown("***")
    st.write("Nicolas has just started his carreer. His father is a good friend of Marin's uncle (they play paddel together each month in Montmartre) and so he holds the title of co-founder...... #Capitalism #Inequalities")

st.markdown("***")
st.write("🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🤑💎🤑💎🤑💎🤑💎🤑💎🤑")

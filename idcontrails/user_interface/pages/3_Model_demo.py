# Imports & variables
import os
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import tempfile
from idcontrails.user_interface.temporary_plot import *


# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Fraisse Tech - Identifying contrails",
    page_icon = ":strawberry:",
    initial_sidebar_state = 'auto')



# Creating the absolute path for assets
absolute_path_root = '/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/app_assets'


########################################################## Contrail gallery ##########################################################

# Displaying top image with logo
st.image(os.path.join(absolute_path_root, 'fraisse_tech_background.jpg'), caption='Fraisse Techologies Incorporated®')


# Explaining the rules
st.subheader("Ground Rules & Objective")
st.markdown("***")
st.write("Our model was trained to identify contrails on satellite images - the same type you just discovered in our gallery!")
st.write("So now let's get into a little competition: YOU👨‍🦰👩‍🦰 vs THE MACHINE🤖!!!")
st.write("We are going to display satellite images from a validation set - that the model hasn't yet seen - and you will have to identify the contrails in it.")
st.write("CAREFUL: there can be images with no contrail !!!")
st.write("Once you are done, the model will have a go and predict the contrails on the sat image. We'll then compare your performances.")
st.markdown("***")



# bet_choice_button = ""
# bet_choice_value = ""
# if started:
#     with st.form("bet_choice"):
#         st.write("🤖:")
#         st.text("Hey there! You are challenging me apparently... How about a friendly wager?")
#         bet_choice_value = st.selectbox("Bet or no bet?", ["Select", "Yes", "No"])
#         bet_choice_button = st.form_submit_button('Next')

# if started is True and bet_choice_button is True and bet_choice_value == "Yes":
#     with st.form("bet_choice_yes"):
#         st.write("🤖:")
#         st.text("Okay we're ooooon")
#         bet_amount = st.select_slider("Place your bet - We're talking € here:", list(range(1,101)), key='bet_amount')
#         bet_amount_button = st.form_submit_button('Next')

# if started is True and bet_choice_button is True and bet_choice_value == "No":
#     with st.form("bet_choice_no"):
#         st.write("🤖:")
#         st.text("A bit of a coward huh? Let's go still")
#         bet_amount_button = st.form_submit_button('Next')



# Launching the game with interactive sequence

# with st.form("start_form"):
#     st.write("Shall we go then?")
#     started = st.form_submit_button("Let's go !")

# Initializing session_state
if 'start_of_choice' not in st.session_state:
    st.session_state['start_of_choice'] = "Nothing"
if 'choice_of_bet' not in st.session_state:
    st.session_state['choice_of_bet'] = "Nothing"
if 'bet_amount' not in st.session_state:
    st.session_state['bet_amount'] = "Nothing"
if 'amout_of_bet' not in st.session_state:
    st.session_state['amout_of_bet'] = -1

st.write(st.session_state['start_of_choice'])
st.write(st.session_state['choice_of_bet'])
st.write(st.session_state['bet_amount'])
st.write(st.session_state['amout_of_bet'])

st.markdown("***")

st.write("Shall we go then?")
started_choice = st.radio("Let's go!", ("Select", "Go!", "Wait"))
st.session_state['start_of_choice'] = started_choice

if st.session_state['start_of_choice'] == "Go!":
    st.write("🤖:")
    st.text("Hey there! You are challenging me apparently... How about a friendly wager?")
    bet_choice = st.selectbox("Bet or no bet?", ["Select", "Yes", "No"])
    st.session_state['choice_of_bet'] = bet_choice

    if st.session_state['choice_of_bet'] == "Yes":
        st.write("🤖:")
        st.text("Okay we're ooooon")
        bet_amount = st.select_slider("Place your bet - We're talking € here:", list(range(1,101)))
        st.session_state['amout_of_bet'] = bet_amount
        st.markdown("***")
        if bet_amount >=0:
            # Loading the data
            uploaded_files = st.file_uploader("Choose input images",
                                    type = ['npy'],
                                    accept_multiple_files=True)
            st.write(len(uploaded_files))

            X_pred = 1
            X_pred_image = 3
            y_true_image = 2

    elif st.session_state['choice_of_bet'] == "No":
        st.write("🤖:")
        st.text("A bit of a coward hu? Let's go still")
        st.markdown("***")
        # Loading the data
        uploaded_files = st.file_uploader("Choose input images",
                                type = ['npy'],
                                accept_multiple_files=True)

        st.write(len(uploaded_files))

        X_pred = 1
        X_pred_image = 3
        y_true_image = 2


    # # Displaying X_pred_image


    # Calling the model to launch a prediction
    url = 'https://www.marinsetraine.com/stagiaire'
    params = {}
    model = requests.get(url, params=params)

    y_pred = model.predict(X_pred)

    # Displaying X_pred_image, the y_pred mask and the combination of both

    # Displaying X_pred_image, the y_true mask and the combination of both

    # Returning the wager result



####################### For reference #######################


# if started:
#     st.write("🤖:")
#     st.text("Hey there! You are challenging me apparently... How about a friendly wager?")
#     bet_choice = st.selectbox("Bet or no bet?", ["Select", "Yes", "No"])
#     if bet_choice == "Yes":
#         st.session_state['choice_of_bet'] = "Yes"
#         st.write(st.session_state['choice of_bet'])

    # if st.session_state.bet_choice == "Yes":
    #     st.session_state.bet_choice = choice_of_bet
    #     st.write("🤖:")
    #     st.text("Okay we're ooooon")
    #     bet_amount = st.select_slider("Place your bet - We're talking € here:", list(range(1,101)), key='bet_amount')
    #     if bet_amount:
    #         st.session_state.bet_amount = bet_amount
    # elif st.session_state.bet_choice == "No":
    #     st.session_state.bet_choice = choice_of_bet
    #     st.write("🤖:")
    #     st.text("A bit of a coward hu? Let's go still")


    # if bet_choice == "Yes":
    #     # st.session_state.bet_choice = bet_choice
    #     st.write("🤖:")
    #     st.text("Okay we're ooooon")
    #     bet_amount = st.select_slider("Place your bet - We're talking € here:", list(range(1,101)), key='bet_amount')
    #     if bet_amount:
    #         st.session_state.bet_amount = bet_amount
    # elif bet_choice == "No":
    #     # st.session_state.bet_choice = bet_choice
    #     st.write("🤖:")
    #     st.text("A bit of a coward hu? Let's go still")














# button_status_1 = st.button("Hello world", disabled=False)
# st.write(f"Button status 1 is {button_status_1}")

# button_status_2 = st.button("Say again", disabled=False)
# st.write(f"Button status 2 is {button_status_2}")

# checkbox_status = st.checkbox("Test check", disabled=False)
# st.write(f"Checkbox status is {checkbox_status}")

# st.radio("Test radio", ["1", "2", "3"])

# st.selectbox("Test select box", ["1", "2", "3"])

# st.multiselect("Test multi-select box", ["1", "2", "3"])

# st.text_input("What do you want to do?")
# st.text_input("Enter password", type="password")

# with st.spinner("Loading ongoing"):
#     f = st.file_uploader("Choose the input image",
#                      type = ['npy'],
#                      accept_multiple_files=True)
# if f is not None:
#     image = getvalue(f)
#     predict on image

# st.balloons()

st.markdown("***")
st.write("🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥")

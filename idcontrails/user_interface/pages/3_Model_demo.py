# Imports & variables
import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display
import plotly.express as px
import requests
import tempfile

from app_processing import *
from idcontrails.ml_logic.plotting_contrails import plot_results, plot_results_streamlit
from idcontrails.interface.main import api_call_predict


# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Fraisse Tech - Identifying contrails",
    page_icon = ":strawberry:",
    initial_sidebar_state = 'auto')



# Creating the absolute path for assets
absolute_path_root_old = '/home/nberretti/code/marinfraisse/identifying-contrails/idcontrails/user_interface/app_assets'
absolute_path_root = os.path.join(os.path.dirname(__file__), "../","app_assets" )


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
    st.session_state['amout_of_bet'] = 1
if 'prediction_start' not in st.session_state:
    st.session_state['prediction_start'] = "Nothing"
if 'display_result' not in st.session_state:
    st.session_state['display_result'] = "Nothing"
if 'prediction_array' not in st.session_state:
    st.session_state['prediction_array'] = np.nan



st.markdown("***")

st.write("Shall we go then?")
started_choice = st.radio("Let's go!", ("Select", "Go!", "Wait"))
st.session_state['start_of_choice'] = started_choice

if st.session_state['start_of_choice'] == "Go!":
    st.write("🤖:")
    st.text("Hey there! You are challenging me apparently... How about a friendly wager?")
    bet_choice = st.selectbox("Bet or no bet?", ["Select", "Bet", "No bet"])
    st.session_state['choice_of_bet'] = bet_choice

    if st.session_state['choice_of_bet'] == "Bet":
        st.write("🤖:")
        st.text("Okay we're ooooon!!!")
        bet_amount = st.select_slider("Place your bet - We're talking € here:", list(range(1,101)))
        st.session_state['amout_of_bet'] = bet_amount

        ############## Sequence if user chose  to bet ###############
        if bet_amount != 1:
            # To be re-used as is in the bet negative
            # Loading the data from the local files
            st.markdown("***")
            st.write("Now let's choose and build an image")
            uploaded_files = st.file_uploader("Choose input images",
                                    type = ['npy'],
                                    accept_multiple_files=True)


            # // Ideally wrap in a function somewhere
            # Retrieving only the 3 bands we want and the true mask
            if uploaded_files:
                st.success("✅ Files loaded")

                for uploaded_file in uploaded_files:
                    if uploaded_file.name == "band_11.npy":
                        band11 = np.load(uploaded_file)
                    elif uploaded_file.name == "band_14.npy":
                        band14 = np.load(uploaded_file)
                    elif uploaded_file.name == "band_15.npy":
                        band15 = np.load(uploaded_file)
                    elif uploaded_file.name == "human_pixel_masks.npy":
                        y_true = np.load(uploaded_file)


                # Ideally put in params
                # Defining bounds for each band and sequence
                N_TIMES_BEFORE = 4
                _T11_BOUNDS = (243, 303)
                _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
                _TDIFF_BOUNDS = (-4, 2)


                # Normalizing the selected image to plot it in RGB ash
                r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
                g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
                b = normalize_range(band14, _T11_BOUNDS)
                image_full_sequences = np.clip(np.stack([r, g, b], axis=2), 0, 1)


                # Choosing only the 5th sequence to plot the image
                input_image = image_full_sequences[..., N_TIMES_BEFORE]

                # Artificially adding 1 dimension to feed to the model
                # X_pred = tf.expand_dims(input_image, 0)
                X_pred = input_image


                # Displaying X_pred_image
                st.markdown("***")
                st.write("Here is the challenge image, with enhanced colors to make it easier for you to see.")
                st.write("Can you spot any contrails?")
                fig = px.imshow(input_image)
                # fig.update_layout(width=400, height=400)
                st.plotly_chart(fig, use_container_width=True)



                # Button to go to the next_step
                st.write("🤖:")
                st.text("Should I have a go at it now?")
                prediction_choice = st.radio("Prediction time!", ("Select", "Predict!", "We're still looking"))
                st.session_state['prediction_start'] = prediction_choice

                if st.session_state['prediction_start'] == "Predict!":
                    st.markdown("***")
                    if type(st.session_state['prediction_array']) == float:
                        with st.spinner('Prediction ongoing...'):
                            # Model call API
                            # Calling the model to launch a prediction
                            y_pred = api_call_predict(X_pred)
                            st.session_state['prediction_array'] = y_pred
                            st.success("Prediction done!")

                    # Building the images
                    y_pred = st.session_state['prediction_array']
                    y_true_image = y_true[..., 0]
                    y_pred_image = y_pred[..., 0]
                    final_layer_true = np.clip(np.stack([r[..., 0]+y_true_image, g[..., 0], b[..., 0]], axis=2), 0, 1)
                    final_layer_pred = np.clip(np.stack([r[..., 0]+y_pred_image, g[..., 0], b[..., 0]], axis=2), 0, 1)


                    # Button to display the results
                    st.write("The model ran, let's look at the results 🥁🥁🥁")
                    result_display = st.radio("Display results?", ("Select", "Show us!", "Wait, it's scary"))
                    st.session_state['display_result'] = result_display
                    st.balloons()


                    # Displaying the results
                    if st.session_state['display_result'] == "Show us!":
                        # Displaying X_pred_image, the y_pred mask and the combination of both
                        # Displaying X_pred_image, the y_true mask and the combination of both
                        # fig = plot_results(input_image, y_true_image, y_pred_image)
                        # st.write(fig)
                        st.markdown("***")
                        plot_results_streamlit(input_image, y_true_image, y_pred_image, final_layer_true, final_layer_pred)

                        # Returning the wager result
                        st.write("🤖:")
                        st.text("Sorry, looks like you just lost some cash 💸💸💸")


        ############## Sequence if user chose NOT to bet ###############

    if st.session_state['choice_of_bet'] == "No bet":
        st.write("🤖:")
        st.text("A bit of a coward hu? Let's go still")

        # Loading the data from the local files
        st.markdown("***")
        st.write("Now let's choose and build an image")
        uploaded_files = st.file_uploader("Choose input images",
                                type = ['npy'],
                                accept_multiple_files=True)


        # // Ideally wrap in a function somewhere
        # Retrieving only the 3 bands we want and the true mask
        if uploaded_files:
            st.success("✅ Files loaded")

            for uploaded_file in uploaded_files:
                if uploaded_file.name == "band_11.npy":
                    band11 = np.load(uploaded_file)
                elif uploaded_file.name == "band_14.npy":
                    band14 = np.load(uploaded_file)
                elif uploaded_file.name == "band_15.npy":
                    band15 = np.load(uploaded_file)
                elif uploaded_file.name == "human_pixel_masks.npy":
                    y_true = np.load(uploaded_file)


            # Ideally put in params
            # Defining bounds for each band and sequence
            N_TIMES_BEFORE = 4
            _T11_BOUNDS = (243, 303)
            _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
            _TDIFF_BOUNDS = (-4, 2)


            # Normalizing the selected image to plot it in RGB ash
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)
            image_full_sequences = np.clip(np.stack([r, g, b], axis=2), 0, 1)


            # Choosing only the 5th sequence to plot the image
            input_image = image_full_sequences[..., N_TIMES_BEFORE]

            # Artificially adding 1 dimension to feed to the model
            # X_pred = tf.expand_dims(input_image, 0)
            X_pred = input_image


            # Displaying X_pred_image
            st.markdown("***")
            st.write("Here is the challenge image, with enhanced colors to make it easier for you to see.")
            st.write("Can you spot any contrails?")
            fig = px.imshow(input_image)
            # fig.update_layout(width=400, height=400)
            st.plotly_chart(fig, use_container_width=True)



            # Button to go to the next_step
            st.write("🤖:")
            st.text("Should I have a go at it now?")
            prediction_choice = st.radio("Prediction time!", ("Select", "Predict!", "We're still looking"))
            st.session_state['prediction_start'] = prediction_choice

            if st.session_state['prediction_start'] == "Predict!":
                st.markdown("***")
                if type(st.session_state['prediction_array']) == float:
                    with st.spinner('Prediction ongoing...'):
                        # Model call API
                        # Calling the model to launch a prediction
                        y_pred = api_call_predict(X_pred)
                        st.session_state['prediction_array'] = y_pred
                        st.success("Prediction done!")

                # Building the images
                y_pred = st.session_state['prediction_array']
                y_true_image = y_true[..., 0]
                y_pred_image = y_pred[..., 0]
                final_layer_true = np.clip(np.stack([r[..., 0]+y_true_image, g[..., 0], b[..., 0]], axis=2), 0, 1)
                final_layer_pred = np.clip(np.stack([r[..., 0]+y_pred_image, g[..., 0], b[..., 0]], axis=2), 0, 1)


                # Button to display the results
                st.write("The model ran, let's look at the results 🥁🥁🥁")
                result_display = st.radio("Display results?", ("Select", "Show us!", "Wait, it's scary"))
                st.session_state['display_result'] = result_display
                st.balloons()


                # Displaying the results
                if st.session_state['display_result'] == "Show us!":
                    # Displaying X_pred_image, the y_pred mask and the combination of both
                    # Displaying X_pred_image, the y_true mask and the combination of both
                    # fig = plot_results(input_image, y_true_image, y_pred_image)
                    # st.write(fig)
                    st.markdown("***")
                    plot_results_streamlit(input_image, y_true_image, y_pred_image, final_layer_true, final_layer_pred)

                    # Returning the wager result
                    st.write("🤖:")
                    st.text("Your were wise not to put your money on the line 🦾")



st.markdown("***")
st.write("🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥🔍🖥")



   ############## FOR REFERENCE - Sequence if user chose NOT to bet ###############

    # if st.session_state['choice_of_bet'] == "No bet":
    #     st.write("🤖:")
    #     st.text("A bit of a coward hu? Let's go still")

    #     # Loading the data from the local files
    #     st.markdown("***")
    #     st.write("Now let's choose and build an image")
    #     uploaded_files = st.file_uploader("Choose input images",
    #                             type = ['npy'],
    #                             accept_multiple_files=True)


    #     # // Ideally wrap in a function somewhere
    #     # Retrieving only the 3 bands we want and the true mask
    #     if uploaded_files:
    #         st.success("✅ Files loaded")

    #         for uploaded_file in uploaded_files:
    #             if uploaded_file.name == "band_11.npy":
    #                 band11 = np.load(uploaded_file)
    #             elif uploaded_file.name == "band_14.npy":
    #                 band14 = np.load(uploaded_file)
    #             elif uploaded_file.name == "band_15.npy":
    #                 band15 = np.load(uploaded_file)
    #             elif uploaded_file.name == "human_pixel_masks.npy":
    #                 y_true = np.load(uploaded_file)


    #         # Ideally put in params
    #         # Defining bounds for each band and sequence
    #         N_TIMES_BEFORE = 4
    #         _T11_BOUNDS = (243, 303)
    #         _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    #         _TDIFF_BOUNDS = (-4, 2)


    #         # Normalizing the selected image to plot it in RGB ash
    #         r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    #         g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    #         b = normalize_range(band14, _T11_BOUNDS)
    #         image_full_sequences = np.clip(np.stack([r, g, b], axis=2), 0, 1)


    #         # Choosing only the 5th sequence to plot the image
    #         input_image = image_full_sequences[..., N_TIMES_BEFORE]

    #         # Artificially adding 1 dimension to feed to the model
    #         X_pred = tf.expand_dims(input_image, 0)


    #         # Displaying X_pred_image
    #         st.markdown("***")
    #         st.write("Here is the challenge image, with enhanced colors to make it easier for you to see.")
    #         st.write("Can you spot any contrails?")
    #         fig = px.imshow(input_image)
    #         # fig.update_layout(width=400, height=400)
    #         st.plotly_chart(fig, use_container_width=True)


    #         # Button to go to the next_step
    #         st.write("🤖:")
    #         st.text("Should I have a go at it now?")
    #         prediction_choice = st.radio("Prediction time!", ("Select", "Predict!", "We're still looking"))
    #         st.session_state['prediction_start'] = prediction_choice

    #         if st.session_state['prediction_start'] == "Predict!":
    #             st.markdown("***")
    #             with st.spinner('Prediction ongoing...'):
    #                 # Model call API

    #                 # Calling the model to launch a prediction
    #                 y_pred = api_call_predict(X_pred)

    #             st.success("Prediction done!")


    #             # Building the images
    #             y_true_image = y_true[..., 0]
    #             y_pred_image = y_pred[..., 0]
    #             final_layer_true = np.clip(np.stack([r[..., 0]+y_true_image, g[..., 0], b[..., 0]], axis=2), 0, 1)
    #             final_layer_pred = np.clip(np.stack([r[..., 0]+y_pred_image, g[..., 0], b[..., 0]], axis=2), 0, 1)


    #             # Button to display the results
    #             st.write("The model ran, let's look at the results 🥁🥁🥁")
    #             result_display = st.radio("Display results?", ("Select", "Show us!", "Wait, it's scary"))
    #             st.session_state['display_result'] = result_display
    #             st.balloons()
    #             st.markdown("***")

    #             # Displaying the results
    #             if st.session_state['display_result'] == "Show us!":
    #                 # Displaying X_pred_image, the y_pred mask and the combination of both
    #                 # Displaying X_pred_image, the y_true mask and the combination of both
    #                 # fig = plot_results(input_image, y_true_image, y_pred_image)
    #                 # st.write(fig)
    #                 plot_results_streamlit(input_image, y_true_image, y_pred_image, final_layer_true, final_layer_pred)

    #                 # Returning the wager result
    #                 st.write("🤖:")
    #                 st.text("Your were wise not to put your money on the line 🦾")

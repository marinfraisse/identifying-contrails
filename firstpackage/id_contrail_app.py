import streamlit as st
import matplotlib_inline as plt
from PIL import Image, ImageOps

# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Fraisse Tech - Identifying contrails",
    page_icon = ":strawberry:",
    initial_sidebar_state = 'auto'
)



# Objective is to display several pages
#  1 - Context and explanations ✅
#  2 - Our process and how we built the app
#  3 - Model demo and Spot the Difference
#  4 - Contrail gallery
#  5 - About us (incl. team) ✅


########################################################## Context page ##########################################################

# Displaying top image with logo
st.image('app_assets/fraisse_tech_background.jpg', caption='Fraisse Techologies Incorporated®')


st.title("Identifying contrails to reduce global warming")
st.write("")

st.header("First up: what are contrails?")
st.write("Contrails are clouds of ice crystals that form in aircraft engine exhaust. Contrails - short for ‘condensation trails’ - are line-shaped clouds of ice crystals that form in aircraft engine exhaust, and are created by airplanes flying through super humid regions in the atmosphere.")
st.write("")


col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.image("app_assets/contrails_left_to_right.jpg", caption="Plane fart")

with col_2:
    st.image("app_assets/sunset_contrails.jpg", caption="A beautiful danger")

with col_3:
    st.image("app_assets/contrails_top_gun.jpg", caption="Tom Cruise making contrails")

st.markdown("***")

st.header("What does it have to do with global warming?")
st.write("Contrails contribute to global warming by trapping heat in the atmosphere. They block heat that normally is released from the earth overnight.  You can consider them as temporary greenhouse clouds that retain heat in the atmosphere. The issue is that with constant air traffic, this temporary effect becomes almost permanent, with huge environmental impact!")
st.write("Contrary to general belief, emissions from burning kerosene do not constitute the bulk of aviation's emissions. Persistent contrails contribute as much to global warming as the fuel that is burnt for flights! After years of research it is now well accepted that contrails contribute approximately 1% of all human caused global warming.")
st.write("")
st.image("app_assets/contrail_stats.png", caption="Source: Google Research, BCG Analysis 😂🎯")
st.markdown("***")

st.header("Keeping both feet on the ground: how can we leverage DL to mitigate the impact from contrails?")
st.write("Contrail avoidance is potentially one of the most scalable, cost-effective sustainability solutions available to airlines today.")
st.write("Researchers have developed models to predict when contrails will form and how much warming they will cause. However, they need to validate these models with satellite imagery. With reliable verification, pilots can have confidence in the models and the airline industry can have a trusted way to measure successful contrail avoidance.")
st.write("The endgame is to have models sufficiently reliable to empower and incetivize airlines to adapt their routes based on each flights risk to generate contrails, based on weather conditions.")
st.markdown("***")

st.write("✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈✈")



########################################################## Contrail gallery ##########################################################

# Displaying top image with logo
st.image('app_assets/fraisse_tech_background.jpg', caption='Fraisse Techologies Incorporated®')

st.title("Contrail Gallery")
st.write("")
st.write("Our U-Net model was trained using satellite images, with a superposition of layers with various bandwiths allowing to capture different types of information.")
st.write("")
st.write("Here is a sample of what the training material looks like:")
st.markdown("***")

# Creating a figure and subplot space for the gallery
# Could add a plotly dynamic visualization by adjusting the number of bands from 1 to 3


images = "list of pictures with the 3 bands and normalized - To be determined and replaced"
gallery_size = "Number of pictures to display"

plt.figure(figsize = (18,8))

for i in range(0, gallery_size):
    plt.subplot(1, gallery_size, i+1)
    plt.imshow(images[i], interpolation='none')


st.write("📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹📷📹")


########################################################## Team and value page ##########################################################

# Displaying top image with logo
st.image('app_assets/fraisse_tech_background.jpg', caption='Fraisse Techologies Incorporated®')

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
    st.image('app_assets/marin_round.png', width=200)
    st.header("Marin Fraisse")
    st.subheader("Founder & CEO")
    st.markdown("***")
    st.write("Marin started his carreer as a strategy consultant at BCG in Paris because he was craving for impact and wanted to apply the rigorous and robust methods he acquired in his scientific studies. #Striving4Greatness #Science")


with column_2:
    st.image('app_assets/robin_round.png', width=200)
    st.header("Robin Diligent")
    st.subheader("Founder & COO")
    st.markdown("***")
    st.write("Robin started his carreer as a strategy consultant at BCG. He has been appointed COO because he has very strong arms (according to a former colleague) and will thus carry the team. #KOstaud #NoPainNoGains")

with column_3:
    st.image('app_assets/nico_round.png', width=200)
    st.header("Nico Berretti")
    st.subheader("Founder & Intern")
    st.markdown("***")
    st.write("Nicolas has just started his carreer. His father is a good friend of Marin's uncle (they play paddel together each month in Montmartre) and so he holds the title of co-founder...... #Capitalism #Inequalities")

st.markdown("***")
st.write("🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🚀💸🤑💎🤑💎🤑💎🤑💎🤑💎🤑")

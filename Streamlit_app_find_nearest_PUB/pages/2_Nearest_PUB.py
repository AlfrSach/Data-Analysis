import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import image
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import os

# Set page configuration
st.set_page_config(
    page_title='Nearest Pubs',
    page_icon=':classical_building:',
    layout='wide',
    initial_sidebar_state='auto'
)

# Main title
st.title(":orange[Find Nearest Pubs!] 🍺🍺")

# Set the path to the resources directory
RESOURCES_PATH = os.path.join(os.path.dirname(__file__),os.pardir, "resources")

# Image
IMAGE_PATH = os.path.join(RESOURCES_PATH,"images", "pubs_near.jpg")
img = image.imread(IMAGE_PATH)
st.image(img,use_column_width=True)

DATA_PATH = os.path.join(RESOURCES_PATH,"data", "cleaned.csv")
df = pd.read_csv(DATA_PATH)

# Get user's location
st.subheader("Enter your location:")
user_lat = st.number_input("Enter your latitude:")
user_lon = st.number_input("Enter your longitude:")

def find_nearest_pubs(df,user_lat,user_lon):

    # Calculate distances and get nearest pubs
    distances = np.sqrt((df['latitude'] - user_lat) ** 2 + (df['longitude'] - user_lon) ** 2)
    df['distance'] = distances
    df = df.sort_values(by=['distance'])
    nearest_pubs = df.head(5)

    # create a map centered at the user's location
    pub_map = folium.Map(location=[user_lat, user_lon], zoom_start=12)

    # add markers for each nearest pub location
    mc = MarkerCluster()
    for row in nearest_pubs.iterrows():
        popup_text = f"<b>{row[1]['name']}</b><br>{row[1]['address']}"
        mc.add_child(folium.Marker(location=[row[1]['latitude'], row[1]['longitude']], popup=popup_text))
    pub_map.add_child(mc)

    # display the map
    st.subheader("Nearest Pubs:")
    if nearest_pubs.empty:
        st.write("No Pubs found in the area.")
    else:
        st.write("Number of Pubs in the area:", nearest_pubs.shape[0])
        folium_static(pub_map)

# create a button to trigger the function
if st.button("Find Nearest Pubs"):
    find_nearest_pubs(df,user_lat,user_lon)
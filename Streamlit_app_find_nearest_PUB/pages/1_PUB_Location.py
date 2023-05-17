import streamlit as st
import pandas as pd
from matplotlib import image
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import os

# Set page configuration
st.set_page_config(
    page_title='Pub Locations',
    page_icon=':classical_building:',
    layout='wide',
    initial_sidebar_state='auto'
)

# Main title
st.title(":orange[Pub Locations!] 🍺🍺")
st.write(":violet[Click on local authority under the location tab and then choose the locality you would like to search ]")

# Set the path to the resources directory
RESOURCES_PATH = os.path.join(os.path.dirname(__file__),os.pardir, "resources")

# Image
IMAGE_PATH = os.path.join(RESOURCES_PATH,"images", "pub_location.jpg")
img = image.imread(IMAGE_PATH)
st.image(img,use_column_width=True)

DATA_PATH = os.path.join(RESOURCES_PATH,"data", "cleaned.csv")
df = pd.read_csv(DATA_PATH)

# select the location type
location_type = st.selectbox(
    "Select the location type:",
    ('Postal Code', 'Local Authority')
)

# filter the dataframe based on the selected location type and location
if location_type == 'Postal Code':
    location_options = df['postcode'].unique()
    location = st.selectbox("Select the Postal Code:", location_options)
    pubs_in_area = df.loc[df['postcode'] == location]
elif location_type == 'Local Authority':
    location_options = df['local_authority'].unique()
    location = st.selectbox("Select the Local Authority:", location_options)
    pubs_in_area = df.loc[df['local_authority'] == location]

# display the number of pubs found in the area
if not pubs_in_area.empty:
    st.write(f"Number of Pubs in the {location_type.lower()} {location} area:", pubs_in_area.shape[0])
else:
    st.write("No Pubs found in the area.")

# create a map centered at the location
if not pubs_in_area.empty:
    pub_map = folium.Map(location=[pubs_in_area['latitude'].mean(), pubs_in_area['longitude'].mean()], zoom_start=12)

    # add markers for each pub location
    mc = MarkerCluster()
    for index, row in pubs_in_area.iterrows():
        popup_text = f"{row['name']}<br>Address: {row['address']}"
        mc.add_child(folium.Marker(location=[row['latitude'], row['longitude']], popup=popup_text))
    pub_map.add_child(mc)

    # display the map
    folium_static(pub_map)
    st.write("Click on the markers to see more information about each pub.")
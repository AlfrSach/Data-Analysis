import streamlit as st
import pandas as pd
from matplotlib import image
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(
    page_title='Open Pubs Application',
    page_icon=':beer:',
    layout='wide',
    initial_sidebar_state='auto'
)

st.title(":orange[Nearest PUB in your locality!:beer:]")
st.subheader(":violet[If you are on a vacation in the United Kingdom and looking for pubs to visit, then you are at the right place.]")
st.write(":violet[I have created a web application that allows you to easily find pubs near you.]")
st.write(":violet[Below you will find the dataset used and also information about top pubs]")
st.write(":violet[Navigate through the dashboard on the left to find the pubs you desire :wave:]")

bt = st.button("Cheers")
if bt == True:
    st.subheader(":beer:")

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources")

# Image
IMAGE_PATH = os.path.join(RESOURCES_PATH,"images","pub.jpeg")
img = image.imread(IMAGE_PATH)
st.image(img)

#Dataset
DATA_PATH = os.path.join(RESOURCES_PATH,"data", "cleaned.csv")

st.header(":green[DataFrame]")
df = pd.read_csv(DATA_PATH)
st.dataframe(df)

st.subheader(":violet[Top 15 Locations :] ")

col1, col2 = st.columns(2)

top_15_locations = df.local_authority.value_counts().head(15).sort_values()
colors = sns.color_palette("viridis", len(top_15_locations))
fig1 = px.bar(top_15_locations, x=top_15_locations.index, y=top_15_locations, labels={"x": "Pubs", "y": "Location"}, title="Top 15 Locations with Most Pubs")

col1.plotly_chart(fig1, use_container_width=True)

txt = "Enjoy the PUBS"
col2 = st.text(txt)












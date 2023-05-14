import streamlit as st
import pickle
import numpy as np
import pandas as pd

#importing model
pipe = pickle.load(open('pipe.pkl','rb'))
cleaned_df = pickle.load(open('cleaned_df.pkl','rb'))

st.title("Laptop Price Predictor")

#Brand
Brand = st.selectbox("Brand",cleaned_df["Brand"].unique())

#Ram Size
Ram_Size = st.selectbox("Ram Size",cleaned_df["Ram Size"].unique())

#Ram Type
Ram_Type = st.selectbox("Ram Type",cleaned_df["Ram Type"].unique())

#Processor
Processor = st.selectbox("Processor",cleaned_df["Processor"].unique())

#HDD
HDD = st.selectbox("HDD(in GB)",[0,128,256,512,1024,2048])

#SSD
SSD = st.selectbox("SSD(in GB)",[0,128,256,512,1024,2048])

# Display
Display = st.selectbox("Display",cleaned_df["Display"].unique())
#Display = st.selectbox("Display(in inches)",['14 inch', '15.6 inch', '17.3 inch', '16 inch', '13 inch', '13.5 inch', '15 inch', '11.6 inch', '16.6 inch'])

#Predict

if st.button("Predict"):
    query = pd.DataFrame([[Brand, Ram_Size, Ram_Type, Processor, HDD, SSD, Display]],columns=['Brand', 'Ram Size', 'Ram Type', 'Processor', 'HDD', 'SSD', 'Display'])

    st.title(float(np.exp(pipe.predict(query)[0])))

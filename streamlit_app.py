import streamlit as st
import os
import pandas as pd
from PIL import Image

dir = './dataSources/'
labels = os.listdir(dir + '/Garbage classification/Garbage classification/')

st.title('Garbage-Classification ML4B')

with st.sidebar:
    st.markdown("[Team Presentation](#team-presentation)", unsafe_allow_html=True)
    st.markdown("[Projekt Presentation](#projekt-presentation)", unsafe_allow_html=True)
  #  st.markdown("[Image classification](#image-classification)", unsafe_allow_html=True)


with st.expander("Team Presentation"):
    st.write("Hi, we are Yannick Rudolf, Nico Schunk and Christoph Lehr. We are creating this app as part of our Machine Learning for Business course.")

with st.expander("Project Presentation"):
    st.write("This app will specify images from six different garbage categories. To categorize them we will use Machine Learning and Deep Learing Techniques. We got the following samples:")

    dicts = []

    for label in labels:
        directory = os.path.join(dir + '/Garbage classification/Garbage classification/', label)
        samples = os.listdir(directory)

        x = {'Type': label, 'Samples': len(samples)}
        dicts.append(x)


    df = pd.DataFrame.from_dict(dicts)
    st.dataframe(df)
    st.write("Our aim is to get uploaded images specified.")

print(labels)
selected_images = st.sidebar.selectbox("Trash Type", labels)

available_images = []
x = os.path.join(dir + 'Garbage classification/Garbage classification/', selected_images)
y = os.listdir(x)
z = st.sidebar.selectbox("Select Sample", y)
a = os.path.join(x,z)
image = Image.open(a)
st.image(image, caption=z)

file = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])

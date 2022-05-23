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
    col1, col2, col3 = st.columns(3)
with col1:
    # Yannick Rudolf
    st.markdown("Yannick Rudolf", unsafe_allow_html=True)
with col2:
    # Nico Schunk
    st.markdown("Nico Schunk", unsafe_allow_html=True)
with col3:
    # Christoph Lehr
    st.markdown("Christoph Lehr", unsafe_allow_html=True)

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
st.header("Samples")
type_path = os.path.join(dir + 'Garbage classification/Garbage classification/', selected_images)
list_of_images = os.listdir(type_path)
image_box = st.sidebar.selectbox("Select Sample", list_of_images)
sample_path = os.path.join(type_path,image_box)
image = Image.open(sample_path)
st.image(image, caption=image_box)

file = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])


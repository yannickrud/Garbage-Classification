import streamlit as st
import os
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


dir = './dataSources/'
labels = os.listdir(dir + '/Garbage classification/Garbage classification/')

st.title('Garbage-Classification ML4B')


with st.expander("Team Presentation"):
    "Hi, we are Yannick Rudolf, Nico Schunk and Christoph Lehr. We are creating this app as part of our Machine Learning for Business course."
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



'''
This app is a tutorial: How to build your own computer vision model following the CRISP-DM Process Model. 

The example used for this tutorial is a garbage classification problem. The data used for this problem can be found on kaggle:
https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
'''

st.camera_input('Welche Art von Müll bist du? Mache jetzt den Test!')
file = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])


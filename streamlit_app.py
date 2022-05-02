import streamlit as st
import os
import pandas as pd

st.title('Garbage-Classification ML4B')

with st.sidebar:
    st.markdown("[Team Presentation](#team-presentation)", unsafe_allow_html=True)
    st.markdown("[Projekt Presentation](#projekt-presentation)", unsafe_allow_html=True)
    st.markdown("[Image classification](#image-classification)", unsafe_allow_html=True)

# TODO - Team Presentation
st.header('Team Presentation')
    print('Hi, we are Yannick Rudolf, Nico Schunk and Christoph Lehr. We are creating this app as part of our Machine Learning for Business course.')
# TODO - Projekt Presentation
st.header('Projekt Presentation')
    
# TODO - Element from Dataset
st.header('Image Classification')

dir = './dataSources/'

labels = os.listdir(dir + '/Garbage classification/Garbage classification/')

col1, col2 = st.columns(2)

dicts = []

for label in labels:
    directory = os.path.join(dir + '/Garbage classification/Garbage classification/', label)
    samples = os.listdir(directory)

    x = {'Type': label, 'Samples': len(samples)}
    dicts.append(x)


df = pd.DataFrame.from_dict(dicts)
st.dataframe(df)

# TODO - Interactive Element

import streamlit as st
import os

st.title('Garbage-Classification ML4B')

# TODO - Team Presentation

# TODO - Projekt Presentation

# TODO - Element from Dataset
dir = './dataSources/'

for file in os.listdir(dir):
    if os.path.isfile(dir + file):
        st.markdown('**' + file + '**')
        st.write(open(dir + file, 'r').read())

# TODO - Interactive Element

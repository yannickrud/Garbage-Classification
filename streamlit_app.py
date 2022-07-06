import streamlit as st
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from PIL.Image import Resampling
import numpy as np
import pathlib

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
dir = './dataSources/'





data_dir = pathlib.Path(os.path.join(dir, './Garbage classification/Garbage classification/'))

st.title('Tutorial: Garbage-Classification ML4B')


with st.expander("Team Presentation"):
    "Hi, we are Yannick Rudolf, Nico Schunk and Christoph Lehr. We are creating this app as part of our Machine Learning for Business course."
    col1, col2, col3 = st.columns(3)

width = 635
height = 845
with col1:
    # Yannick Rudolf
    img = Image.open('images/Yannick.jpg')
    resized_img = img.resize((width,height))
    st.image(resized_img)
    st.markdown("Yannick Rudolf", unsafe_allow_html=True)
with col2:
    # Nico Schunk
    img = Image.open('images/Nico.jpg')
    resized_img = img.resize((width, height))
    st.image(resized_img)
    st.markdown("Nico Schunk", unsafe_allow_html=True)
with col3:
    # Christoph Lehr
    img = Image.open('images/Christoph.jpg')
    resized_img = img.resize((width, height))
    st.image(resized_img)
    st.markdown("Christoph Lehr", unsafe_allow_html=True)



'''
This app is a tutorial: How to build your own computer vision model following the CRISP-DM Process Model. 


The example used for this tutorial is a garbage classification problem. The data used for this problem can be found on kaggle:
https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
###
'''

with st.expander("CRISP-DM-Model"):

    img = Image.open('images/CRISPDMModel.png')
    st.image(img)
    "The Cross industry standard process for data mining (CRISP-DM process model) gives you a guide that helps you creating an ordered approach towards building, understanding," \
    "modeling, evaluation and deploying your buisness idea based on a dataset."
'''
###
'''
img = st.camera_input('What kind of trash are you? Go ahead and take the test!')


model = tf.keras.models.load_model('./model/cv_model.h5')


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

if img:
    class_names = train_ds.class_names
    img = Image.open(img)
    img = img.resize((img_width, img_height), Resampling.LANCZOS)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.convert_to_tensor(img_array[:, :, :3])

    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    'This taken image most likely belongs to',class_names[np.argmax(score)],' with a ', 100 * np.max(score),'percent confidence.'

'''
###
'''
file = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])


if file:
    class_names = train_ds.class_names
    img = Image.open(file)
    img = img.resize((img_width, img_height), Resampling.LANCZOS)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.convert_to_tensor(img_array[:, :, :3])

    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    'This uploaded image most likely belongs to',class_names[np.argmax(score)],' with a ', 100 * np.max(score),'percent confidence.'

'''
###
### Samples
'''
selected_images = st.selectbox("Trash Type", os.listdir(data_dir))
type_path = os.path.join(data_dir, selected_images)
list_of_images = os.listdir(type_path)
image_box = st.selectbox("Select Sample", list_of_images)
sample_path = os.path.join(type_path,image_box)
image = Image.open(sample_path)
st.image(image, caption=image_box)
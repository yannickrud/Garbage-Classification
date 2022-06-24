import streamlit as st
import tensorflow as tf
import pathlib
import os

dir = './dataSources/'
data_dir = pathlib.Path(os.path.join(dir, './Garbage classification/Garbage classification/'))


st.set_page_config(page_title="Data preparation")

'''
# Data preparation
Data preparation is divided into 5 steps in the CRISP-DM process model.

1. Select data
2. Clean data
3. Construct data
4. Integrate data
5. Format data

'''

'''
## Select data
When training a model, there is a possibility that the data set is biased. Therefore, care must be taken during 
development to keep the data set as bias-free as possible. In this example the image "plastic1.jpg" is a water bottle.
There is no serious damage recognizable which could be interpreted as the bottle can still be used. Although some of those
examples exists in the dataset we declare all data provided as garbage.
'''

validation_split = st.slider(label='Select the size of the validation data set',min_value=0.01, max_value=0.99, step=0.01)

batch_size = 32
img_height = 384
img_width = 512

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

'''
## Clean data
The data is in such good condition that it does not need to be cleaned at this time. Since modeling and data preparation
are quite closely related, data may still need to be cleaned.
'''

'''
## Construct data
'''

'''
## Integrate data
'''

'''
## Format data
'''
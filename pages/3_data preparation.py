import streamlit as st
import tensorflow as tf
import pathlib
import os
from tensorflow import keras
from tensorflow.keras import layers, callbacks

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

The data has to be split into training and validation data. Common splits are: 80% test, 20% validation; 67% Train, 33% Test

You can play around and look how the different train/test splits influence the performance of the model.
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

class_names = train_ds.class_names

col1, col2, col3 = st.columns(3)

for images, labels in train_ds.take(1):
  for i in range(9):
    if i % 3 == 0:
      with col1:
        st.image(caption=class_names[labels[i]], image=images[i].numpy().astype("uint8"))
    elif i % 3 == 1:
      with col2:
        st.image(caption=class_names[labels[i]], image=images[i].numpy().astype("uint8"))
    elif i % 3 == 2:
      with col3:
        st.image(caption=class_names[labels[i]], image=images[i].numpy().astype("uint8"))


'''
## Clean data
The data is in such good condition that it does not need to be cleaned at this time. Since modeling and data preparation
are quite closely related, data may still need to be cleaned.
'''

'''
## Construct data

Now we apply data augmentation to our data
'''

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

'''
## Integrate data
'''

'''
## Format data
'''
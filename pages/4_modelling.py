import streamlit as st

st.set_page_config(page_title="Modelling")

'''
# Modelling
'''

'''
## Select Modeling Technique
The first step in modeling is to select the specific modeling technique.
#### Modeling Technique 
The actual modeling technique we used is Convolutional Neural Network (CNN), a neural network has multiple filter kernels per convolutional layer
, creating layers of feature maps that each get the same input but extract different features due to different weight matrices. 
'''
st.code('''
import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
''')
'''
#### Modeling Assumptions
The assumptions coming from this model are a required size. This problem is easy fixable since you can simply resize the images if they dont have
the proper size.
'''
st.code('''
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
''')


'''
## Generate test design
'''
'''
Before we create our final model, we need to check the quality and accuracy with the help of a test model.
For this we provide the function with the necessary datasets and specify how many epochs the model should run through.
'''
st.code('''
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[early_stopping])
''')

'''
## Build model
'''

'''
## Asses model
'''
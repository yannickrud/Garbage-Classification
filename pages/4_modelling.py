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
The assumptions coming from this model are a required size. This problem is easy fixable since you can simply resize the images if they dont have
the proper size.


## Build model
'''
st.code('''
num_classes = len(class_names)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])''')
'''
- "layers.Rescaling(...) rescales the height and width of the image"
- "layers.Conv2D(...) creates a convolution kernel (3x3 in this case) that is convolved with the layer input to produce a tensor of outputs (16)."
- "layers.MaxPooling2D() is much like a Conv2D layer, except that it uses a simple maximum function instead of a kernel, with the pool_size parameter analogous to kernel_size." \
- " A MaxPool2D layer doesn't have any trainable weights like a convolutional layer does in its kernel, however."
- "layers.Conv2D() [with higher filter values] and layers.MaxPooling2D() get repeated to get more precise results."
- "layers.Dropout() is a layer that prevents the model from overfitting by randomly setting input units to 0 by the frequency rate of (0.2)."
- "layers.Flatten() flattens the input, but does not affect the batch size. It converts two dimensional outputs of the base into the one dimensional inputs needed by the head."
- "layers.Dense(128,...) performs a layer of hidden units."
- "layers.Dense(num_classes) transforms the output to a probability score for the different trash classes."
'''
st.code('''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
''')
'''

## Generate test design
'''
'''
Before we create our final model, we need to check the quality and accuracy with the help of a test model.
For this we provide the function with the necessary datasets and specify how many epochs the model should run through.
Incrementing the epochs makes the model more and more accurate up to a certain point.
However, this is additionally dependent on the learning rate.
If it is very low, the model learns too slowly.
If it is too high, the model forgets what it has learned before and starts more or less from the beginning.
Therefore, the learning rate must be selected appropriately.
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
As an addition the option early-stopping can be added. This causes the model to stop as soon as the validation accuracy becomes constantly worse.



'''

'''
## Asses model
'''
st.image("images/modelResult1.png")
'''
Shown here are the results of the model after using 10 epochs, our model has an accuracy of approx. 70% on the test data sample and 
an accuracy of 64% on the validation data sample.
So there is probably still room for improvements. 
'''

'''
'''
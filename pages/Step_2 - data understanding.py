from PIL import Image
import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Data understanding")

'''
# Data understanding 
Data understanding can be divided into 4 Steps.

1. Collect initial data
2. Describe data
3. Explore data
4. Verify data quality

## Collect initial data
In this section the target is to acquire and load your data in your preferred tool.

The data for this project is provided on kaggle:
https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

First we load the labels of the pictures in a pandas dataframe with each category.
'''

import pathlib

dir = './dataSources/'
data_dir = pathlib.Path(os.path.join(dir, './Garbage classification/Garbage classification/'))

dicts = []

for label in os.listdir(data_dir):
    directory = os.path.join(data_dir, label)
    samples = os.listdir(directory)

    for img in samples:
        z = Image.open(os.path.join(directory, img))
        x = {'img_name': img, 'category': label, 'size': z.size}
        dicts.append(x)

df = pd.DataFrame.from_dict(dicts)

st.code('''
dir = './dataSources/'
data_dir = pathlib.Path(os.path.join(dir, './Garbage classification/Garbage classification/'))

dicts = []

for label in os.listdir(data_dir):
    directory = os.path.join(data_dir, label)
    samples = os.listdir(directory)

    for img in samples:
        z = Image.open(os.path.join(directory, img))
        x = {'img_name': img, 'category': label, 'size': z.size}
        dicts.append(x)

df = pd.DataFrame.from_dict(dicts)
''', language='python')



'''
## Describe data

The dataset contains classified pictures of garbage. The categories are cardboard, glass, metal, paper, plastic and trash.

'''
number_of_samples = df[['img_name', 'category']].groupby('category').count()
number_of_samples

'The total number of samples: ', df['img_name'].count()
'''
## Explore data
### Samples
'''

selected_images = st.selectbox("Trash Type", os.listdir(data_dir))
type_path = os.path.join(data_dir, selected_images)
list_of_images = os.listdir(type_path)
image_box = st.selectbox("Select Sample", list_of_images)
sample_path = os.path.join(type_path,image_box)
image = Image.open(sample_path)
st.image(image, caption=image_box)

'The size of the pictures:'
size_values = df['size'].unique().tolist()
for item in size_values:
    x = {'width':item[0], 'height': item[1]}
    x
'''
## Verify data quality

After sampling the data the classification is correct. The pictures are all the same size.
The amount of data is enough to train a model. With data augmentation the training data for the model can be extended
which leads to better results. This has to be done carefully. For example a car picture should not be flipped horizontal
because this could lead to poorer performance of the models prediction.
'''
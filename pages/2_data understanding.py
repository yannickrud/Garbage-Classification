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

dir = './dataSources/'
labels = os.listdir(dir + '/Garbage classification/Garbage classification/')

dicts = []

for label in labels:
    directory = os.path.join(dir + '/Garbage classification/Garbage classification/', label)
    samples = os.listdir(directory)

    for img in samples:
        x = {'img': img, 'category': label}
        dicts.append(x)

df = pd.DataFrame.from_dict(dicts)



'''
## Describe data

The dataset contains classified pictures of garbage. The categories are cardboard, glass, metal, paper, plastic and trash.

'''
number_of_samples = df.groupby('category').count()
number_of_samples

'The total number of samples: ', df['img'].count()
'''
## Explore data
### Samples
'''
selected_images = st.selectbox("Trash Type", labels)
type_path = os.path.join(dir + 'Garbage classification/Garbage classification/', selected_images)
list_of_images = os.listdir(type_path)
image_box = st.selectbox("Select Sample", list_of_images)
sample_path = os.path.join(type_path,image_box)
image = Image.open(sample_path)
st.image(image, caption=image_box)

'''
## Verify data quality
'''
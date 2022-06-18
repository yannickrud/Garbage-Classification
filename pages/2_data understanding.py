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

'''





'''
## Describe data
'''

dir = './dataSources/'
labels = os.listdir(dir + '/Garbage classification/Garbage classification/')

dicts = []

for label in labels:
    directory = os.path.join(dir + '/Garbage classification/Garbage classification/', label)
    samples = os.listdir(directory)

    x = {'Type': label, 'Samples': len(samples)}
    dicts.append(x)


df = pd.DataFrame.from_dict(dicts)

st.dataframe(df)

'''
## Explore data
'''

'''
## Verify data quality
'''
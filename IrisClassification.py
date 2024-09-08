
import sys
#sys.path.append('/Volumes/TOSHIBA EXT/Maestria/Programs/Machine Learning/End-to-End Housing/')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle as pkl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from joblib import dump, load




print(sys.path)

# from exports import custom_transformers

#full_pipeline = joblib.load('/Volumes/TOSHIBA EXT/Maestria/Programs/Machine Learning/End-to-End Housing/exports/full_pipeline.pkl')

pipiline_path = '/Volumes/TOSHIBA EXT/GitHub/IrisClassification/exports/pipeline.pkl'
with open(pipiline_path, 'rb') as file1:
    print(file1.read(100))  
try:
    pipeline = joblib.load(pipiline_path)
    print("pipeline loaded successfully!")
except Exception as e:
    print("Failed to load pipeline:", e)

model_path = '/Volumes/TOSHIBA EXT/GitHub/IrisClassification/exports/model.pkl'

with open(model_path, 'rb') as file:
    print(file.read(100))  
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load model:", e)

#print(pipeline)

st.write("""# Iris Classification""")

sepal_length = st.slider('Enter a sepal length (3.0 to 8.0)', min_value=3.0, max_value=8.0, value=5.0, step=0.1, format= '%.1f')
sepal_width = st.slider('Enter a sepal width (1.0 to 5.0)', min_value=1.0, max_value=5.0, value=3.0, step=0.1, format= '%.1f')
petal_length = st.slider('Enter a petal length (1.0 to 8.0)', min_value=1.0, max_value=8.0, value=4.0, step=0.1, format= '%.1f')
petal_width = st.slider('Enter a petal width (0.0 to 3.0)', min_value=0.0, max_value=3.0, value=1.0, step=0.1, format= '%.1f')


col3, col4, col5 = st.columns([1,6,1])

if st.button('Predict'):
    with col4:
        input_data = pd.DataFrame(
    
        {'sepal_length_(cm)': [sepal_length], 'sepal_width_(cm)': [sepal_width], 'petal_length_(cm)': [petal_length], 'petal_width_(cm)': [petal_width]},
         index=[0]
    )
        st.write(input_data)

        pipelined_data = pipeline.transform(input_data)

        prediction = model.predict(pipelined_data)

        if prediction[0] == 0:
            prediction_text = 'Setosa'
        elif prediction[0] == 1:
            prediction_text = 'Versicolor'
        elif prediction[0] == 2:
            prediction_text = 'Virginica'

        col6, col7, col8 = st.columns(3)

        with col7:
            st.markdown('<h2 style="text-align: center;"> The predicted Iris is ' + str(prediction_text) + ' </h2>', unsafe_allow_html=True)
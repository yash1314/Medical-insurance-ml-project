import sys
import os

from src.pipelines.prediciton_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

import streamlit as st
import time
import numpy as np

predict = PredictPipeline()

# Image
st.image('images/istockphoto-868640146-1024x1024.jpg', width=300)

# Title
st.title('Medical Insurance cost predictor')
st.markdown('#### This model can predict Medical charges with an accuracy score of 90%')

# input field
st.markdown("#### Age")
age = st.text_input('')

st.markdown('#### Gender')
gender = st.text_input("Male-1, female-0")

st.markdown("#### BMI")
bmi = st.text_input("Enter BMI value in range of (15-55)")

st.markdown("#### Number of Children")
children = st.text_input("Input number")

st.markdown("#### Smoker")
smoker = st.text_input("Smoke: Yes - 1, No - 0")

st.markdown("#### Region")
region = st.text_input("Region: southwest-0, southeast-1, northwest-2, northeast-3: ")

if st.button('Predict'):
    try:
        input_data = [age, gender, bmi, children, smoker, region]
        
    except Exception as e:
        st.markdown("### Please enter valid data !")
        logging.info('Error occured in app file')
        raise CustomException(e, sys)
        

    else:
        result = np.round(predict.predict(input_data))

        bar = st.progress(50)
        time.sleep(1)
        bar.progress(100)

        st.info('Success')
        st.markdown(f'**Your Predicted health Insurance charge is : $ {result}**')
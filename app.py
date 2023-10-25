import sys
import os

from src.pipelines.prediciton_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

import streamlit as st
import time
import numpy as np

predict = PredictPipeline()


# Title
st.title('Medical Insurance cost predictor')
st.markdown('This model can predict Medical charges with an accuracy score of 90%')

cols1, cols2 = st.columns(2)

with cols1:
# input field
    st.markdown("#### Age")
    age = st.text_input('Age')

    st.markdown('#### Gender')
    gender = st.text_input("Male-1, female-0")

    st.markdown("#### BMI")
    bmi = st.text_input("Enter BMI value in range of (15-55)")

with cols2:
    st.markdown("#### Number of Children")
    children = st.text_input("Input number")

    st.markdown("#### Smoker")
    smoker = st.text_input("Yes - 1, No - 0")

    st.markdown("#### Region")
    region = st.text_input("southwest-0, southeast-1, northwest-2, northeast-3: ")


if st.button('calculate'):
    try:
        input_data = [age, gender, bmi, children, smoker, region]
        
    except Exception as e:
        st.markdown("### Please enter valid data!")
        logging.info('Error occured in app file')
        raise CustomException(e, sys)
        
    else:
        try:
            result = np.round(predict.predict(input_data))
        
        except Exception as e:
            st.markdown("### Please enter valid data!")
            # logging.info('Error occured in app file')
            # raise CustomException(e, sys)
        else:

            with st.spinner('Processing'):
                time.sleep(1)

            st.header(f'Your Predicted health Insurance charge is : $ {result}')
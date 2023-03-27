from ast import operator
import streamlit as st
import cv2
import numpy as np
import pandas as pd

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    # st.experimental_show(dataframe)
    st.write(dataframe)
    
st.text("Chọn Input")
A = st.checkbox("A")
B = st.checkbox("B")
C = st.checkbox("C")
st.text("Output: D")
st.text("Độ đo")
MAE = st.checkbox("MAE")
MSE = st.checkbox("MSE")
Number = st.number_input('Train/Test split')
Kfold = st.checkbox("kfold Cross-Validation")
if Kfold:
    st.number_input("k")
Clicked = st.button("run")

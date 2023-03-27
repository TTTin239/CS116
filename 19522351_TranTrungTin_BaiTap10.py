import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import time
import streamlit as st


def model_compare(data, split_size):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Class']), data['c lass'], test_size=1-split_size)

    #XGBoost model
    xgb = XGBClassifier(n_estimators=100)
    training_start = time.perf_counter()
    xgb.fit(X_train, y_train)
    training_end = time.perf_counter()

    prediction_start = time.perf_counter()
    preds = xgb.predict(X_test)
    prediction_end = time.perf_counter()

    acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
    xgb_train_time = training_end-training_start
    xgb_prediction_time = prediction_end-prediction_start

    #SVC model
    svc = SVC()
    training_start = time.perf_counter()
    svc.fit(X_train, y_train)
    training_end = time.perf_counter()

    prediction_start = time.perf_counter()
    preds = svc.predict(X_test)
    prediction_end = time.perf_counter()

    acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100
    svc_train_time = training_end-training_start
    svc_prediction_time = prediction_end-prediction_start


    #DT model
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    training_start = time.perf_counter()
    clf_gini.fit(X_train, y_train)
    training_end= time.perf_counter()

    prediction_start = time.perf_counter()
    preds = clf_gini.predict(X_test)
    prediction_end = time.perf_counter()
    acc_dt = (preds == y_test).sum().astype(float) / len(preds)*100
    dt_train_time = training_end-training_start
    dt_prediction_time = prediction_end-prediction_start

    results = pd.DataFrame({
    'Model': ['XGBoost', 'SVC', 'DT'],
    'Score': [acc_xgb, acc_svc, acc_dt],
    'Runtime Training': [xgb_train_time, svc_train_time, dt_train_time],
    'Runtime Prediction': [xgb_prediction_time, svc_prediction_time, dt_prediction_time]})
    result_df = results.sort_values(by='Score', ascending=False)
    result_df = result_df.set_index('Model')


    return result_df


#--Streamlit--#
st.title("Model Comparision")

uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=None)

    st.write(data)
    

    split_size = st.slider("Train/Test split", 0.0, 1.0, 0.8)

    result = model_compare(data, split_size)

    if st.button('Result'):
        st.write(result)
    else:
        st.write()
from re import X
from tkinter import Y
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

st.title('Linear Regression')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    st.markdown("**Chọn Input**")
    input = [ 0  for i in range(len(df.columns))]
    col1, col2, col3 = st.columns(3)
    for i in range(len(df.columns) - 2):
        if i == 0:
            with col1:
                input[i] = st.checkbox(f"Column {i+1}")
                if input[i]:
                    st.write(df.iloc[:, i+1])
        elif i == 1:
            with col2:
                input[i] = st.checkbox(f"Column {i+1}")
                if input[i]:
                    st.write(df.iloc[:, i+1])
        else:
            with col3:
                input[i] = st.checkbox(f"Column {i+1}")
                if input[i]:
                    st.write(df.iloc[:, i+1])
    
    st.markdown("**Output**")
    st.write(df.iloc[:, len(df.columns) - 1])
    
    st.markdown("**Độ đo**")
    col4, col5 = st.columns(2)
    with col4:
        MAE = st.checkbox("MAE")
        if MAE:
            pp = 0
    with col5:
        MSE = st.checkbox("MSE")
        if MSE:
            pp = 1

    st.markdown("**Evaluation**")
    col6, col7 = st.columns(2)
    with col6:
        tts = st.checkbox("Train/Test split")
        if tts:
            number_1 = st.number_input('Num of Train/Test split')
    with col7:
        kfold = st.checkbox("K-fold Cross-Validation")
        if kfold:
            number_2 = st.number_input("k")

    def model_train_test_split(X,y,per,pp):
        #model use train test split for evaluation
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=per, random_state=42) 

        lr = LinearRegression()
        lr.fit(X_train,y_train)

        print('Score of LR:',lr.score(X_test,y_test))
        y_pred = lr.predict(X_test)
        y_train_pred = lr.predict(X_train)
        if pp == 0:
            score_train = mean_squared_error(y_train, y_train_pred)
            score_test = mean_squared_error(y_test, y_pred)
        else :
            score_train = mean_absolute_error(y_train, y_train_pred)
            score_test = mean_absolute_error(y_test, y_pred)
        #draw hist for train-test-split
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1)
        ax.bar([0,1],[0,score_train ], color ='r', width= 0.2)
        ax.set_xlabel('number ')
        ax.set_ylabel('training error')
        ax.set_title('Training error across folds')

        ax2 = fig.add_subplot(1,2,2)
        ax2.bar([0,1], [0,score_test], width= 0.2)
        ax2.set_xlabel('number ')
        ax2.set_ylabel('testing error')
        ax2.set_title('Testing error across folds')

        st.pyplot(fig)

    def model_kfold(X,y,n_fold,pp):
        #model use k-fold for evaluation

        kf = KFold(n_splits=int(n_fold))
        kf.get_n_splits(X)
        list_training_error = []
        list_testing_error = []
        # print(kf)
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lr = LinearRegression()
            lr.fit(X_train,y_train)
            y_train_data_pred = lr.predict(X_train)
            y_test_data_pred = lr.predict(X_test) 

            if pp == 1:
                fold_training_error = mean_absolute_error(y_train, y_train_data_pred)        
                fold_testing_error = mean_absolute_error(y_test, y_test_data_pred)
            else :
                fold_training_error = mean_squared_error(y_train, y_train_data_pred)        
                fold_testing_error = mean_squared_error(y_test, y_test_data_pred)
            list_training_error.append(fold_training_error)
            list_testing_error.append(fold_testing_error)
        #draw hist for k-fold
        fig2 = plt.figure()
        ax = fig2.add_subplot(1,2,1)
        ax.bar(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), color ='r')
        ax.set_xlabel('number of fold')
        ax.set_ylabel('training error')
        ax.set_title('Training error across folds')
   
        ax2 = fig2.add_subplot(1,2,2)
        ax2.bar(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel())
        ax2.set_xlabel('number of fold')
        ax2.set_ylabel('testing error')
        ax2.set_title('Testing error across folds')
        st.pyplot(fig2)
    features = []
    for i in range(len(df.columns)):
        if input[i] == 1 :
            features.append(i)

    clicked = st.button("run")
    if clicked:
        a = features[0]
        b = features[1]
        X = df.iloc[:,a].values
        y = df.iloc[:,b].values
        X = X.reshape(-1,1)
        y = y.reshape(-1,1)

        if kfold:
            model_kfold(X,y,number_2,pp) 
        if tts:
            model_train_test_split(X,y,number_1,pp)
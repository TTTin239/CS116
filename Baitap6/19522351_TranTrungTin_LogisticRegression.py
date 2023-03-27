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
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

st.title('Logistic Regression')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,sep = None)
    st.write(df)
    print(df.info())
    st.markdown("**Chọn Input**")
    input = [ 0  for i in range(len(df.columns))]
    col1, col2, col3 = st.columns(3)
    for i in range(2):
        if i == 0:
            with col1:
                input[i] = st.checkbox(f"Column {i+1}")
                if input[i]:
                    st.write(df.iloc[:, i])
        elif i == 1:
            with col2:
                input[i] = st.checkbox(f"Column {i+1}")
                if input[i]:
                    st.write(df.iloc[:, i])
        # else:
        #     with col3:
        #         input[i] = st.checkbox(f"Column {i+1}")
        #         if input[i]:
        #             st.write(df.iloc[:, i])
    
    st.markdown("**Output**")
    st.write(df.iloc[:, len(df.columns) - 1]) 
    st.markdown("**Độ đo**")
    col4, col5 = st.columns(2)
    with col4:
        f1 = st.checkbox("F1 - Score")
        if f1:
            pp = 0
    with col5:
        ll = st.checkbox("Log loss")
        if ll:
            pp = 1

    st.markdown("**Evaluation**")
    col6, col7 = st.columns(2)
    with col6:
        tts = st.checkbox("Train/Test split")
        if tts:
            number_1 = st.number_input('Num of Train/Test split', step=0.1)
    with col7:
        kfold = st.checkbox("K-fold Cross-Validation")
        if kfold:
            number_2 = st.number_input("k", step=1)

    def model_train_test_split(X,y,per,pp):
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=per, random_state=4) 
    
        lr = LogisticRegression()
        lr.fit(X_train,y_train)
        print('Score of LR:',lr.score(X_test,y_test))

        y_pred = lr.predict(X_test)
        y_train_pred = lr.predict(X_train)

        st.text(len(np.unique(y_test)))

        if pp == 0:
            score_train = f1_score(y_train, y_train_pred,average='micro')
            st.text(score_train)
            score_test = f1_score(y_test, y_pred,average='micro')
            st.text(score_test)
        else :
            y_train_pred = lr.predict_proba(X_train)
            score_train = log_loss(y_train, y_train_pred)  
            st.text(score_train)
            y_pred = lr.predict_proba(X_test)    
            score_test = log_loss(y_test,y_pred)
            st.text(score_test)

        fig = plt.figure()
        ax = fig.add_subplot(1,2,1)
        ax.bar([0,1],[0,score_train ], color ='r', width= 0.2)
        ax.set_xlabel('number ')
        ax.set_ylabel('Score')
        ax.set_title('Training score')

        ax2 = fig.add_subplot(1,2,2)
        ax2.bar([0,1], [0,score_test], width= 0.2)
        ax2.set_xlabel('Number ')
        ax2.set_ylabel('Score')
        ax2.set_title('Testing score')

        st.pyplot(fig)

    def model_kfold(X,y,n_fold,pp):
        #model use k-fold for evaluation

        kf = KFold(n_splits=int(n_fold))
        kf.get_n_splits(X)
        list_training_error = []
        list_testing_error = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lr = LogisticRegression()
            lr.fit(X_train,y_train)
            y_train_data_pred = lr.predict(X_train)
            y_test_data_pred = lr.predict(X_test) 

            if pp == 1:
                fold_training_error = log_loss(y_train,lr.predict_proba(X_train))        
                fold_testing_error = log_loss(y_test,lr.predict_proba(X_test))
            else :
                fold_training_error = f1_score(y_train, y_train_data_pred,average='micro')        
                fold_testing_error = f1_score(y_test, y_test_data_pred,average='micro')
            list_training_error.append(fold_training_error)
            list_testing_error.append(fold_testing_error)
        #draw hist for k-fold
        fig2 = plt.figure()
        ax = fig2.add_subplot(1,2,1)
        ax.bar(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), color ='r')
        ax.set_xlabel('number of fold')
        ax.set_ylabel('training score')
        ax.set_title('Training ')
   
        ax2 = fig2.add_subplot(1,2,2)
        ax2.bar(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel())
        ax2.set_xlabel('number of fold')
        ax2.set_ylabel('testing score')
        ax2.set_title('Testing ')
        st.pyplot(fig2)
    features = []
    for i in range(len(df.columns)):
        if input[i] == 1 :
            features.append(i)

    clicked = st.button("Run")
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

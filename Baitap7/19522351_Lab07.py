import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import streamlit as st

def LR_w_PCA(data, input_ft, split_size, PCA_cpn):
    
    target_name = data.columns.values[0]
    target = data[target_name]
    df = data.drop([target_name],axis=1)

    if len(input_ft) != 1:
        X = df[input_ft].values
    else:
        X = [[value] for value in df[input_ft[0]].values]

    X_train,X_test,y_train,y_test = train_test_split(X,target,test_size = 1-split_size,random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    if PCA_cpn != len(input_ft):
        pca = PCA(n_components=PCA_cpn)
        tr_comp = pca.fit_transform(X_train)
        ts_comp = pca.transform(X_test)

        pc_model = LogisticRegression()
        pc_model.fit(tr_comp,y_train)

        report = classification_report(y_test,pc_model.predict(ts_comp), output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        fig = plt.figure()
        plt.scatter(tr_comp[:,0], tr_comp[:,1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        
        return report_df, fig

    else:
        model = LogisticRegression()
        model.fit(X_train,y_train)

        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        report_df = pd.DataFrame(report).transpose()
    
        return report_df



#--Streamlit--#
st.title("Classification with PCA")

uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=None)

    st.write(data)
    
    feature_options = data.columns.values[1:]
    dft = feature_options[1:]


    choosen_ft = st.multiselect(
        "Input feature ",
        options= feature_options,
        default = dft,
    )

    split_size = st.slider("Train/Test split", 0.0, 1.0, 0.8)

    n_options = [i for i in range(2,len(choosen_ft))]

    n_components = st.selectbox(
        "Select n_components for PCA",
        options = n_options,
    )
    if n_components != 0:
        rp, plot = LR_w_PCA(data, choosen_ft, split_size, n_components)
        st.pyplot(plot)
    else:
        rp = LR_w_PCA(data, choosen_ft, split_size, n_components)

    if st.button('Result'):
        st.write(rp)
    else:
        st.write()

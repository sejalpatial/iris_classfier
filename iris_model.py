# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:30:59 2026

@author: Acer
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)
st.markdown("""
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #45a049;
    color: white;
}

</style>
""", unsafe_allow_html=True)

iris_logistic_model=pickle.load(open('iris_model_logistic.sav','rb'))
iris_knn_model=pickle.load(open('iris_model_knn.sav','rb'))
iris_random_forest_model=pickle.load(open('iris_model_random_forest.sav','rb'))

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🌸 Iris Flower Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>Compare Machine Learning Models for Iris Flower Prediction</h4>",
    unsafe_allow_html=True
)

st.divider()

with st.sidebar:
    st.image("iris.png", width=200)
    st.title("Navigation")
    selected =option_menu('Different models for flower classification',
                          ['Logistic Regression','KNN','RandomForest','model comparison'],
                          default_index=0)

if selected in ['Logistic Regression','KNN','RandomForest']:
    st.markdown("### Enter Flower Details")
    col1,col2 = st.columns(2)
    with col1:
        sepal_length=st.slider("Sepal Length(cm)",min_value=4.0,
        max_value=8.0,
        value=5.8,
        step=0.1)
    
        sepal_width=st.slider("Sepal Width(cm)",
                     min_value=2.0,
                     max_value=4.5,
                     value=3.0,
                     step=0.1)
    with col2:
        petal_length=st.slider("Petal Length(cm)",
                     min_value=1.0,
                     max_value=7.0,
                     value=5.0,
                     step=0.1)
        petal_width=st.slider("Petal Width(cm)",
                     min_value=0.1,
                     max_value=2.50,
                     value=1.0,
                     step=0.1)
    input_data=[[sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width]]
    
     
    if (selected=='Logistic Regression'):
          
            if st.button('Prediction using Logistic Regression'):
                pred=iris_logistic_model.predict(input_data) 
                probability = iris_logistic_model.predict_proba(input_data)
                st.info("### 🌼 Prediction")
                st.success(f"🌸 {pred[0]}")
                st.subheader("Prediction Confidence")

                confidence_df = pd.DataFrame({
                    "Species": iris_logistic_model.classes_,
                    "Confidence (%)": [f"{p*100:.2f}%" for p in probability[0]]
                })
            
                st.table(confidence_df)
            
    if (selected=='KNN'):
           
            if st.button('Prediction using KNN'):
                pred=iris_knn_model.predict(input_data)
                probability = iris_knn_model.predict_proba(input_data)
                st.info("### 🌼 Prediction")
                st.success(f"🌸 {pred[0]}")
                st.subheader("Prediction Confidence")

                confidence_df = pd.DataFrame({
                    "Species": iris_knn_model.classes_,
                    "Confidence (%)": [f"{p*100:.2f}%" for p in probability[0]]
                })
            
                st.table(confidence_df)
                
    if (selected=='RandomForest'):
            
            if st.button('Prediction using Random Forest'):
                pred=iris_random_forest_model.predict(input_data)
                probability = iris_random_forest_model.predict_proba(input_data)
                st.info("### 🌼 Prediction")
                st.success(f"🌸 {pred[0]}")
                st.subheader("Prediction Confidence")

                confidence_df = pd.DataFrame({
                    "Species": iris_random_forest_model.classes_,
                    "Confidence (%)": [f"{p*100:.2f}%" for p in probability[0]]
                })
            
                st.table(confidence_df)
            

if(selected=='model comparison'):
    st.header(" Model Performance Comparison")
 

    models = ['Logistic Regression', 'KNN', 'Random Forest']
    accuracies = [0.9667, 1.0, 0.9667]
    
    plt.figure(figsize=(8,5))
    plt.bar(models, accuracies)
    plt.ylim(0.90,1.05)

    plt.grid(axis='y', linestyle='--')
    plt.title("Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    
    for i in range(len(models)):
        plt.text(i,
                 accuracies[i] + 0.01,
                 f"{accuracies[i]:.2f}",
                 ha='center')
    
    st.pyplot(plt)
    st.metric(
    label="🏆 Best Model",
    value="KNN",
    delta="Accuracy : 100%"
)

    
        
        
        
        
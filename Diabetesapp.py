# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:04:07 2020

@author: zini
"""
# To run the app open comand prompt: 'conda activate dp' and type 
# 'streamlit run Diabetesapp.py' from the file directory
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#import time

st.write("""
# Simple Diabetes Prediction App
This app predicts the likelihood of having **Diabetes**!
""")
#This app outputs the **Diabetes** prediction performance
#""")

# =============================================================================
#-------------------------------------------
@st.cache
def loadataset(url):
    return pd.read_csv(url)

url ="https://raw.githubusercontent.com/Philosfin/Deploy/master/diabetes.csv"

df = loadataset(url)  #pd.read_csv(url, error_bad_lines=False) #('diabetes.csv')

X=df.drop('Outcome', axis=1)
y=df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

if st.checkbox('Show summary description of training dataset'):
    st.subheader('Decribe the training dataset:')
    st.write(X_train.describe())


st.subheader('Input parameters')
st.write(X_test)

if st.checkbox('Show summary description of test dataset'):
    st.subheader('Decribe the test dataset:')
    st.write(X_test.describe())

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

st.subheader('AUC:')
st.write(roc_auc_score(y_test, y_pred_prob))

#=============================================================================
# my_bar = st.progress(0)
# for complete_percent in range(100): 
#     my_bar.progress(complete_percent + 1)
#     time.sleep(0.01)
#=============================================================================
cv_auc = cross_val_score(logreg, X_train, y_train, scoring = 'roc_auc', cv=5)
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))   
    
    
st.subheader('AUC scores computed using 5-fold cross-validation:')
st.write(cv_auc)
#c_space = np.logspace(-5, 8, 15)
param_grid = {'C': np.linspace(1,50)} #arange(1,50)}


logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X_train, y_train)

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

st.subheader('Tuned Logistic Regression Parameters:')
st.write(logreg_cv.best_params_)
st.subheader('Best score is:')
st.write(logreg_cv.best_score_)

prediction = logreg_cv.predict(X_test)
prediction_proba = logreg_cv.predict_proba(X_test)[:,1]

st.subheader('Tuned parameter Cross-validated AUC:')
st.write(roc_auc_score(y_test, prediction_proba))
#--------------------------------------------

st.subheader('Prediction')
st.write(prediction[:])

st.subheader('Prediction Probability')
st.write(prediction_proba[ : ])

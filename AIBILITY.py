#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import random
import numpy as np
import xlrd


# In[2]:


data = pd.read_excel("P:/COB_F.xls")


# In[3]:


train_data = data.drop(["Unnamed: 0","Claim ID","Primary","Secondary","Plan_Type","CoInsurance","OOP","Primary_paid","Secondary_paid","TPL"], axis=1)


# In[4]:


from sklearn import preprocessing
 
label_encoder = preprocessing.LabelEncoder()
train_data['Claim Type']= label_encoder.fit_transform(train_data['Claim Type'])


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
X = train_data.drop("Total_Coverage", axis=1)
y = train_data["Total_Coverage"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[6]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[7]:


from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as sm
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[8]:


st.markdown("<h1 style='text-align: center; color: White;background-color:#FF7F7F'>AIbility Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: Black;'>Drop in The required Inputs and we will do  the rest.</h3>", unsafe_allow_html=True)
st.sidebar.header("What is this Project about?")
st.sidebar.text("It a Web app that would help the user in determining their Total Coverage for the Claim.")
st.sidebar.header("What tools where used to make this?")
st.sidebar.text("The Model was made using a dataset from CMS along with using Jupyter notebooks to train the model. We made use of Sci-Kit learn in order to make our Random Forest Regression Model.")
Claim_Type = st.selectbox('Please select the Claim Type',('Hospital Inpatient Claims', 'Long Term Care Claims', 'Medicare Part A Claims','Medicare Part B Claims','Out Patient Claims','Professional Service Claims','RHC, FQHC, IHC Claims'))
#part of our main method
Member_ID = st.number_input("Input Your Member ID")
Claim_ID = st.number_input("Input Your Claim ID") 
Billed_amt = st.number_input("Input Billed Amount")


if (Claim_Type=='Hospital Inpatient Claims'):
    ctype=0
elif (Claim_Type=='Long Term Care Claims'):
    ctype=1
elif (Claim_Type=='Medicare Part A Claims'):
    ctype=2   
elif (Claim_Type=='Medicare Part B Claims'):
    ctype=3
elif (Claim_Type=='Out Patient Claims'):
    ctype=4
elif (Claim_Type=='Professional Service Claims'):
    ctype=5
elif (Claim_Type=='RHC, FQHC, IHC Claims'):
    ctype=6
if st.button("Predict"):

    allowed_amt = np.random.randint(1000,3000,1)
    coins_list =[0,20,25,30,35,40,45,50]
    coinsp = random.choice(coins_list)
    coin = (coinsp*Billed_amt)/100
    Secondary_allowed_amt = np.random.randint(500,3500,1)
    deduc = np.random.randint(30,70,1)
    copay=np.random.randint(0,100,1)


    new_input = [[ctype,Billed_amt,allowed_amt,deduc,copay,Secondary_allowed_amt,coin]]
    new_input = np.array(new_input)
    new_input = np.reshape(new_input,(new_input.shape[0],new_input.shape[1]))
    new_input = sc.transform(new_input)
    pred1 = regressor.predict(new_input)
    st.subheader('Total Out Of Pocket Expenses to be paid is {}'.format(copay[0]+coin+deduc[0]))
    st.subheader('Total Coverage of your Claim is {}'.format(round(pred1[0],2)))


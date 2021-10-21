#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np



# In[2]:



 
# Give the location of the file
loc = "P:/COB.xlsx"
 

data = pd.read_excel(loc)


# In[3]:


train_data = data.drop(["Unnamed: 0","Claim ID","Primary","Secondary","Plan_Type","CoInsurance","OOP","Primary_paid","Secondary_paid","TPL"], axis=1)


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
X = train_data.drop("Total_Coverage", axis=1)
y = train_data["Total_Coverage"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[5]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[6]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)


# In[7]:


from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as sm
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[8]:


'''
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 Score:', metrics.r2_score(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
'''


# In[9]:


import random
billed_amt = np.random.randint(4000,6000,1)
allowed_amt = np.random.randint(1000,3000,1)
coins_list =[0,20,25,30,35,40,45,50]
coinsp = random.choice(coins_list)
coin = (coinsp*allowed_amt)/100
Secondary_allowed_amt = np.random.randint(500,3500,1)
deduc = np.random.randint(30,70,1)
copay=np.random.randint(100,500,1)


new_input = [[billed_amt,allowed_amt,deduc,copay,Secondary_allowed_amt,coin]]
new_input = np.array(new_input)
new_input.shape
new_input = np.reshape(new_input,(new_input.shape[0],new_input.shape[1]))


# In[10]:


new_input = sc.transform(new_input)


# In[11]:


pred1 = regressor.predict(new_input)


# In[12]:



Mem = st.text_input("Enter your member ID ")
Bill = st.text_input("Enter your claim amount")
p_class = st.selectbox("Plan Type",options=['Plan1' , 'Plan2' , 'Plan3'])
st.subheader('Your total coverage is {}'.format(pred1))


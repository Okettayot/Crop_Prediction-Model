#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


# In[73]:


# Load the data from a CSV file
data = pd.read_csv('weather_crop_data.csv')


# In[74]:


print(data)


# In[75]:


# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Varieties of Crops grown'])


# In[76]:


df = pd.DataFrame


# In[77]:


df = pd.DataFrame(data)


# In[78]:


pd.DataFrame(data)


# In[79]:


# Split the data into training and testing sets
Varieties = ['Varieties of Crops grown_Cassava','Varieties of Crops grown_Green Peas','Varieties of Crops grown_Kales','Varieties of Crops grown_Maize','Varieties of Crops grown_Sorghum','Varieties of Crops grown_Sunflower','Varieties of Crops grown_Sweet Potatoes','Varieties of Crops grown_spinach']
X = data.drop(Varieties, axis=1)
y = data[Varieties]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[80]:


# Train a random forest classifier on the training set
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


# In[81]:


# Evaluate the model on the testing set
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[82]:


# Predict the crop type for a given month
month = pd.DataFrame({
    'Max Tempt 2019': [35],
    'Min Tempt 2019': [20],
    'Max Tempt 2020': [33],
    'Min Tempt 2020': [21],
    'Max Tempt 2021': [32],
    'Min Tempt 2021': [22],
    'Rainfall 2019': [50],
    'Rainfall 2020': [70],
    'Rainfall 2021': [60],
    'Months':[12]
})
crop_type = rf.predict(month)
print('Predicted crop type:', crop_type[0])


# In[84]:


# Save the trained model to a file
joblib.dump(crop_type, 'crop_prediction_model.joblib')


# In[ ]:





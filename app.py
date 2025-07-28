import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import pandas as pd
import pickle
import keras
from keras._tf_keras.keras.models import load_model
#from tensorflow.keras_models import load_model
#load models

#model = tf.keras_models.load_models('model.keras')
model = load_model('model.keras')
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_loaded= pickle.load(file)

with open("data_encoder_geo.pkl","rb") as file:
    data_encoder_geo_loaded= pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler= pickle.load(file)

st.title("Customer Churn Prediction")

#user_input

geography=st.selectbox('Geography',data_encoder_geo_loaded.categories_[0])
encoded_values = [0, 1]
gender_labels = label_encoder_loaded.inverse_transform(encoded_values)
gender=st.selectbox('Gender',gender_labels)
age=st.slider('Age', 18, 92)
tenure=st.slider('Tenure', 0, 10)
credit_Score=st.number_input('Credit Score')
estimatedSalary=st.number_input('Estimated Salary')
balance=st.number_input('Balance')
numberofProducts=st.slider('Number Of Products', 1, 4)
hasCrCard=st.selectbox('Has Credit Card',[0,1])
isActiveMember=st.selectbox('Is an Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_Score],
    'Gender':[label_encoder_loaded.fit_transform([gender])[0]],
    'Age' : [age],
    'Tenure' :[tenure],
    'Balance': [balance],
    'NumOfProducts':[numberofProducts],
    'HasCrCard':[hasCrCard], 
    'IsActiveMember':[isActiveMember],
    'EstimatedSalary':[estimatedSalary],
})
#
geo_encoded= data_encoder_geo_loaded.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=data_encoder_geo_loaded.get_feature_names_out(['Geography']))
input_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_scaled= scaler.transform(input_df)
prediction=model.predict(input_scaled)
prediction_prob =(prediction[0][0])
st.write("prediction probability value:",prediction_prob)
if prediction_prob>0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
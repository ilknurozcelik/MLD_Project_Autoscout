
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import Lasso
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
import pickle

   
st.markdown("<h1 style='text-align:center; color:black;'>Car Price Prediction</h2>", unsafe_allow_html=True)
df =  pd.read_csv("final_df.csv")

# Adding image
st.image("https://thinkingneuron.com/wp-content/uploads/2020/09/Car-price-prediction-case-study.png", width=700)


# add button
if st.checkbox("Show Data") :
    st.write(df.head())
    
# add warning
st.error(":point_left:Please input the features of car you interested **using sidebar**, before making price prediction!!!")

st.sidebar.title("Please select features of car you interested")

# Collects user input features into dataframe
def user_input_features() :
    make_model = st.sidebar.selectbox("Model Type", df["make_model"].unique())
    Gearing_Type = st.sidebar.selectbox("Gearing Type", df["Gearing_Type"].unique())
    Gears = st.sidebar.selectbox("Gears",(5,6,7,8))
    age = st.sidebar.number_input("Car Age",min_value=0, max_value=3)
    hp_kW = st.sidebar.slider("Horse Power (kW)", 40.0, 239.0, 50.0, 1.0)
    km = st.sidebar.slider("Kilometers (km)", 0.0, 317000.0, 5000.0, 100.0)
      
    
    data = {"make_model" : make_model,
            "Gearing_Type" : Gearing_Type,
            "age" : age,
            "hp_kW" : hp_kW,
            "km" : km,
            "Gears" : Gears}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Read the saved model
model= pickle.load(open("final_model_scout", "rb"))

# Apply model to make predictions
if st.button('Predict Car Price'):
    st.success(f'Predicted Car Price:&emsp;${model.predict(input_df)[0].round(2)}')

st.markdown("Thank you for visiting our **Car Price Prediction** page.")
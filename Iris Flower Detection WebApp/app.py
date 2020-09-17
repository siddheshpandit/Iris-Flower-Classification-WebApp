import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
import pickle

iris=datasets.load_iris()
pickle_in=open('kmeans.pkl','rb')
kmeans=pickle.load(pickle_in)


st.title("Iris Flower Classification")
html_temp="""
<div style="background-color:teal ; padding:10px ">
<h2 style="color:white;text-align:center;">Iris Flower Classification</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

def input():
    sl=st.slider("Select Sepal Length",0.0,8.0)
    sw=st.slider('Select Sepal Width',0.0,8.0)
    pl=st.slider("Select Petal Length",0.0,8.0)
    pw=st.slider('Select Petal Width',0.0,8.0)

    data={"Sepal Length":sl,"Sepal Width":sw,"Petal Length":pl,"Petal Width":pw}
    inputs=pd.DataFrame(data,index=[0])
    return inputs

user_inputs=input()
st.write(user_inputs)

output=kmeans.predict(user_inputs)
st.subheader("Prediction")

st.write(iris.target_names[output])
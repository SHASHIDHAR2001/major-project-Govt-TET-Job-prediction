import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.metrics import accuracy_score
d = pd.read_csv("data.csv")
cat_dict={}
dist_dict={}
sub_dict={}
st.markdown("""<h2 style='color:green'>Government Teacher Job Selection Predictor</h2>""",unsafe_allow_html=True)
key=list(d['Category'].unique())
dist=list(d['District'].unique())
sub=list(d['Subject'].unique())
label_encoding = preprocessing.LabelEncoder()
d['Category'] = label_encoding.fit_transform(d['Category'])
cat_dict=dict(zip(key,d['Category'].unique()))
d['Gender'] = label_encoding.fit_transform(d['Gender'])
d['District'] = label_encoding.fit_transform(d['District'])
dist_dict=dict(zip(dist,d['District'].unique()))
d['Subject'] = label_encoding.fit_transform(d['Subject'])
sub_dict=dict(zip(sub,d['Subject'].unique()))
x= d.iloc[:,3:8].values
y = d.iloc[:,-1].values
clf = lgb.LGBMClassifier()
clf.fit(x,y)
Y = clf.predict(x)
accuracy=accuracy_score(y, Y)
def app():
        name=st.text_input("Enter your name")
        roll_num=st.text_input("Enter your Roll Number")
        gender=st.selectbox("Choose Gender", options=[" ","Female","Male"])
        if gender=="Female":
            gender=0
        if gender=="Male":
            gender=1
        category=st.selectbox("Choose suitable Category", options=[" "]+key)
        if category!=" ":
            category=cat_dict[category]
        merit_score=st.number_input('Insert Merit Score')
        district=st.selectbox("Choose District",options=[" "]+dist)
        if district!=" ":
            district=dist_dict[district]
        subject=st.selectbox("Choose Subject",options=[" "]+sub)
        if subject!=" ":
            subject=sub_dict[subject]
        if st.button("Predict"):
            values=np.array([[gender,category,merit_score,district,subject]])
            try:
                result=clf.predict(values)
            except:
                st.warning("Enter correct Data")
            result='ok'
            if result=='No':
                #st.title("Better Luck next time")
                st.markdown("""<h2 style='color:red'>Better Luck next time</h2>""",unsafe_allow_html=True)
            if result=='Yes':
                #st.title("Congratulations You will be eligible")
                st.markdown("""<h2 style='color:green'>Congratulations You will be eligible</h2>""",unsafe_allow_html=True)
                st.balloons()
if __name__=="__main__":
    app()

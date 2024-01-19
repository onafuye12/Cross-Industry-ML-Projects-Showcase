import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('RF_class_model.pkl','rb'))
scaler = pickle.load(open('scal_class.pkl','rb'))
encoder = pickle.load(open('enc_class.pkl','rb'))



st.header("BANK CHURN PREDICTION")

def data1():
    CreditScore = st.number_input(' What is your credit score')
    Geography = st.selectbox('Where are you located',['France', 'Spain', 'Germany'])
    Gender = st.selectbox('What is your Gender',['Female', 'Male'])
    Age = st.number_input('Input your age')
    Tenure = st.selectbox(' Select tenure ',[ 2,  1,  8,  7,  4,  6,  3, 10,  5,  9,  0])
    Balance = st.number_input('What is your balance')
    NumOfProducts = st.selectbox('What is the NumOfProducts',[1, 3, 2, 4])
    HasCrCard = st.selectbox('Do you have a credit card',[1, 0])
    IsActiveMember = st.selectbox('Are you an active member',[1, 0])
    EstimatedSalary= st.number_input(' What is your estimated salary')
    
   
    feat = np.array([CreditScore, Geography, Gender, Age, Tenure, Balance,
       NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]).reshape(1,-1)
    cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    feat1 = pd.DataFrame(feat, columns=cols)
    
    return feat1
    
frame = data1()


def prepare(df):
   
    enc_data =pd.DataFrame(encoder.transform(df[['Geography', 'Gender']]).toarray())
    #enc_data.columns = encoder.get_feature_names_out()
    enc_data.columns = encoder.get_feature_names(['Geography', 'Gender'])
    df= df.join(enc_data)

    df.drop(['Geography', 'Gender'],axis=1,inplace=True)
    cols = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns=cols)
    return df
    
frame2= prepare(frame)


    
if st.button('predict'):
    #frame2= prepare(frame)
    pred = model.predict(frame2)
    if pred[0] == 0:
        st.write('Not Churn')
    else:
        st.write('Churn')
    #st.write(pred[0])
    
    
    

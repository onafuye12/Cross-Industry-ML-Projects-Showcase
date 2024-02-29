import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('loan_model.pkl','rb'))
scaler = pickle.load(open('scal.pkl','rb'))

st.title('LOAN DEFAULT PREDICTION')
def data1():
    Employed = st.selectbox('Employed',[0,1])
    Bank_Balance = st.number_input(' Bank Balance')
    Annual_Income = st.number_input('Annual Income')
    Monthly_Income = st.number_input('Monthly Income')

    feat = np.array([Employed, Bank_Balance, Annual_Income, Monthly_Income]).reshape(1,-1)
    cols = ['Employed', 'Bank_Balance', 'Annual_Income', 'Monthly_Income']
    
    feat1 = pd.DataFrame(feat, columns=cols)
    
    return feat1
    
frame = data1()
if st.button('Show Input Data'):
    st.write(frame.head())

def prepare(df):
    #df = pd.get_dummies(data=df , columns=['Type'],drop_first=True)
    #df['Type'] = df['Type'].map({'L':0, 'M':1,'H':2})
    cols = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns=cols)
    return df
    
frame2= prepare(frame)

if st.button('Show process Data'):
    st.write(frame2.head())
    
if st.button('predict'):
    #frame2= prepare(frame)
    pred = model.predict(frame2)
    if pred[0] == 0:
        st.write("This individual might not default")
    else:
        st.write("This individual might default")
    #st.write(pred[0])
    #st.write(pred[0])
    

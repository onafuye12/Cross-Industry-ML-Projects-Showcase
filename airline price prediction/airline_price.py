import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from PIL import Image



model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scal.pkl','rb'))
encoder = pickle.load(open('enc.pkl','rb'))

st.header("AIR TRAVEL PRICE PREDICTION ")
#img = Image.open('download.jfif')
#img = img.resize((700,200))
#st.image(img)



def data1():
    Airline = st.selectbox('Select an Airline',['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
       'Vistara Premium economy', 'Jet Airways Business',
       'Multiple carriers Premium economy', 'Trujet'])
    Source = st.selectbox('Select your location',['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    Destination = st.selectbox(' Select your destination',['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    Total_Stops = st.selectbox('Select Types of stop',['non-stop', '2 stops', '1 stop', '3 stops', '4 stops'])
    Duration_hours = st.number_input(' Input estimated time of duration for your flight')
    Duration_minutes = st.number_input(' Input the remaining minutes if it is not a whole hour ')
    Month = st.selectbox('Select the month you will like to travel (january=1.....december=12)',[1,2,3,4,5,6,7,8,9,10,11,12])
    #WeekStatus_Weekend = st.number_input(' Week Status')
        
    feat = np.array([Airline, Source, Destination,Total_Stops,Duration_hours, Duration_minutes, Month]).reshape(1,-1)
    cols = ['Airline', 'Source', 'Destination','Total_Stops','Duration_hours', 'Duration_minutes', 'Month']
    
    feat1 = pd.DataFrame(feat, columns=cols)
    
    return feat1
    
frame = data1()


def prepare(df):
    # select the columns to encode
    columns_to_encode = ['Airline', 'Source', 'Destination', 'Total_Stops']

    # transform the selected columns
    df[columns_to_encode] = encoder.transform(df[columns_to_encode])

    cols = df.columns
    #scaler = MinMaxScaler()
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns=cols)
    return df

frame2= prepare(frame)


    
if st.button('predict'):
    #frame2= prepare(frame)
    pred = model.predict(frame2)
    st.write(pred[0])
    
    
    

from turtle import heading, left
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data 
from keras.models import load_model # we will not do epochs again and again
import streamlit as st 
import requests 
from streamlit_lottie import st_lottie
from PIL import Image


st.set_page_config(page_title="Predict Your Stocks",page_icon=":sunglasses:",layout="wide")

def load_lottieurl(url) : 
    r = requests.get(url)
    if r.status_code != 200 : 
        return None
    return r.json()


lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_w51pcehl.json")
img = Image.open("images/dp.jpeg")

start = '2010-01-01'
end = '2019-12-31'
with st.container() :
    st.subheader("Hi, I am Sourav Kumar :wave:")
    st.title('Stock trend prediction')
    st.write("I have made this app to help you guys predict the stocks ")
    st.write("[Learn More about stocks >](https://finance.yahoo.com/)")

with st.container() :
    st.write("-----")
    left_column,right_column,image_column = st.columns(3)
    with left_column : 
        st.header("About me")
        st.write(
            """
            Hello I am sourav kumar , i have done my schooling from Army public
            school rk puram, Currently i am pursuing Btech in 
            computer Science engineering in Graphic era university.
            my hobbies are coding ,playing football"""
        )
    with image_column :
        st.image(img)

    with right_column :
        st_lottie(lottie_coding,height=300,key="coding")


user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#describing data
st.subheader('Date from 2010-2019')
st.write(df.describe())

#graphs

#closing price 
st.subheader('Closing price vs Time chart ')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

#100 days moving average
st.subheader('Closing price vs time chart with 100 Moving Average')
mavg100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(mavg100)
plt.plot(df.Close)
st.pyplot(fig)

#200 days moving average
st.subheader('Closing price vs time chart with 100 Moving Average and 200 Moving Average')
mavg100 = df.Close.rolling(100).mean()
mavg200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(mavg100)
plt.plot(mavg200,'r')
plt.plot(df.Close)
st.pyplot(fig)

#splitting data into training and testing 

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#just load my model not need to do epochs again here

model = load_model('keras_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index= True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]) :
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])


x_test , y_test = np.array(x_test),np.array(y_test)

#making predictions
y_predict = model.predict(x_test)

scaler = scaler.scale_  # gives us the factor for which it is scaled down so we divide ypred and ytest by this

scale_factor = 1/scaler[0] #as it is different for different stocks
y_predict = y_predict*scale_factor
y_test = y_test*scale_factor

#final output 
st.subheader('Predicted value vs Origianl value')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="original price")
plt.plot(y_predict,'r',label="predicted price")
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)


with st.container() :
    st.write("----")
    st.header("Get In Touch with me !")
    st.write("##")
    contact_form = """
    <form action="https://formsubmit.co/sk2000cs@gmail.com" method="POST">
        <input type = "hidden" name="_captcha" value = "false">
        Enter Your Name : <input type="text" style="background-color:yellow;border=90%" name="name"  required>
        <br>
        <br>
        Enter Your Email : <input type="email" name="email"  required>
        <br>
        <br>
        Enter Your Message : <textarea name="message" required></textarea>
        <br>
        <br>
        <button type="submit">Send</button>
    </form>
    """
    left_column,right_column = st.columns(2)
    with left_column :
        st.markdown(contact_form,unsafe_allow_html = True)
    with right_column : 
        st.empty()

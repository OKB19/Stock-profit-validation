import numpy as np 
import requests
import pandas as pd
from keras.models import load_model
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
from sklearn.model_selection import train_test_split
from flask import render_template


# model = load_model('humanAI.h5') 
def preprocessdata(stock):
    stock = stock.upper()
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-historical-data"
    querystring = {"period1":"1546448400","period2":"1562086800","symbol":stock,"frequency":"1d","filter":"history"}

    headers = {
        "X-RapidAPI-Key": "c042502a7fmshf40ad5370b96568p1e82b0jsn027123d0fcf3",
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        print(response.status_code)
        a =  response.json()
        data = pd.DataFrame(a["prices"])
        if stock =="AAPL" or stock =="MSFT" or stock =="ASML" or data.shape[1]==10:
            data = data.drop(columns=["date","volume","adjclose",'amount','type','data'],axis=1)
        if stock =="TSLA" or data.shape[1]==7:
            data = data.drop(columns=["date","volume","adjclose"],axis=1)
        data = data.dropna()
        x = data.iloc[:,:-1]
        y = data.iloc[:,-1:]
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
        model = keras.Sequential()
        model.add(Dense(5,activation="relu"))
        model.add(Dense(60,activation="relu"))
        model.add(Dense(30,activation="relu"))
        model.add(Dense(12,activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
        model.fit(X_train,y_train,epochs = 20)
        y_pred = model.predict(X_test)
        print(y_pred[0][0])
        original = np.array(y_test)
        if original[0][0] > y_pred[0][0]:
            return 0
        else:
            return 1
    except requests.exceptions.RequestException as e:
        print(e)
        return f'An error occurred: {str(e)}'

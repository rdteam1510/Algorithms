from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import plotly.graph_objects as go
from pymongo import MongoClient
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

client = MongoClient("mongodb+srv://tradingvision:123@cluster0.xmnn8.mongodb.net/TradingVision?authSource=admin&replicaSet=atlas-kkwgbw-shard-0&w=majority&readPreference=primary&appname=MongoDB%20Compass&retryWrites=true&ssl=true")


# Specify date format
# def parser(x):
# 	return datetime.strptime(x,'%Y%m%d')

#Get ticker ' ' 
def sort_data(ticker):
    db = client["ForPrediction"]
    data = db["For_Prediction"]
    df = data.find({"Ticker":ticker}).sort('Time',-1).limit(7)
    return df
# Data Preprocessing
def get_data(get_ticker):
    df = sort_data(get_ticker)
    df = list(df)
    df.reverse()
    close_data = [i['Close'] for i in df]
    close_data = np.array(close_data)
    close_date = [i['Time'] for i in df]
    return close_data, close_date
# close_data = scaler.fit_transform(close_data)
# close_train = scaler.fit_transform(close_train)
# close_test = scaler.transform(close_test)
# Time series generator
def only_train(close_data, model, ticker):
    look_back = 6
    train_generator = TimeseriesGenerator(close_data, close_data, length = look_back, batch_size=20)
    num_epochs = 25
    earlyStopping = EarlyStopping(
        monitor='loss',
        patience=10, 
        verbose=0, 
        mode='min')
    mcp_save = ModelCheckpoint(
        'project\CodeWithMongoDB\Checkpoint\model-{}.h5'.format(ticker), 
        save_best_only=True, 
        monitor='loss', 
        mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='loss', 
        factor=0.1, 
        patience=7, 
        verbose=1, 
        epsilon=1e-4, 
        mode='min')
    model.fit_generator(
        train_generator, 
        epochs=num_epochs, 
        verbose=1, 
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
    return model
def predict(num_prediction, model, close_data):
    prediction_list = close_data[-num_prediction:]
    #prediction_list = np.array([])
    for _ in range(num_prediction-1):
        x = prediction_list[-num_prediction:]
        x = x.reshape((1, num_prediction, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[num_prediction:]
        
    return prediction_list
    
def predict_dates(num_prediction, close_date):
    close_date = [datetime.datetime.fromtimestamp(i) for i in close_date]
    last_date = close_date[-1] + pd.DateOffset(days = 1)
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1)
    # Create a list of Saturday and Sunday 
    leap = []
    for day in prediction_dates:
       if (day.strftime('%A') == "Saturday") | (day.strftime('%A') == "Sunday"):
           leap.append(day)

    # Delete Saturday and Sunday
    prediction_dates = prediction_dates.drop(leap)

    # To list for drawing prediction
    prediction_dates = prediction_dates.tolist()

    return prediction_dates

def make_predict(num_prediction, model, close_data, close_date):
    forecast = predict(num_prediction, model, close_data)
    forecast = [round(elem, 2) for elem in forecast]
    num = ((num_prediction-1)/5-1)*2
    forecast_dates = predict_dates(num_prediction + num, close_date)
    return forecast, forecast_dates

def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

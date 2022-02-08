from keras.models import load_model
from CodeWithMongoDB.train_model import * 
from pymongo import MongoClient

client

if __name__ == "__main__":
    # Choose time and close price columns 
    
    ticker = 'AFX'
    
    # Get data for retraining
    close_data, close_date = get_data(ticker)
    
    # Retrain model
    model = load_model('project\CodeWithMongoDB\Model\{}_model'.format(ticker))
    model = only_train(close_data, model, ticker)
    # Predict price
    num_prediction = 6
    forecast, forecast_dates = make_predict(num_prediction, model, close_data, close_date)

     # Save new model
    #model.save('Model\{}_model'.format(ticker))
    # print("Model exported")
    print(close_data, close_date)
    print(forecast, forecast_dates)
    

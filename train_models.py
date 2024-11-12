import pandas as pd
from lstm_model import train_lstm_model
from gru_model import train_gru_model
from cnn_model import train_cnn_model
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('data/NVDA_daily_stock_data.csv')


lstm_model, lstm_scaler = train_lstm_model(data)
lstm_model.save('../models/saved_models/lstm_nvda_model.h5')

gru_model, gru_scaler = train_gru_model(data)
gru_model.save('../models/saved_models/gru_nvda_model.h5')

cnn_model, cnn_scaler = train_cnn_model(data)
cnn_model.save('../models/saved_models/cnn_nvda_model.h5')

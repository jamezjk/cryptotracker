import numpy as np
import pandas as pd
from datetime import timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
data = pd.read_csv('data/NVDA_daily_stock_data.csv')  # Adjust this to your actual dataset file path

# Set the start date for predictions (one day after the last date in the dataset)
start_date = pd.to_datetime("2024-11-05")

# Number of days to predict (e.g., one week)
prediction_days = 7

# Prepare your data for prediction (scaling and reshaping as needed for your models)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

# Load your models
lstm_model = load_model('../models/saved_models/lstm_nvda_model.h5')
gru_model = load_model('../models/saved_models/gru_nvda_model.h5')
cnn_model = load_model('../models/saved_models/cnn_nvda_model.h5')

# Prepare input data for each model (assuming you need a sequence of the last 60 data points)
sequence_length = 60
input_sequence = scaled_data[-sequence_length:]  # Take the last 60 data points

# Reshape for model input
input_sequence = input_sequence.reshape(1, sequence_length, 1)

# Predict with each model
lstm_predictions = []
gru_predictions = []
cnn_predictions = []

# Generate predictions for each day in the prediction period
for _ in range(prediction_days):
    # Predict with each model
    lstm_pred = lstm_model.predict(input_sequence)
    gru_pred = gru_model.predict(input_sequence)
    cnn_pred = cnn_model.predict(input_sequence)

    # Inverse scale to get actual prices
    lstm_price = scaler.inverse_transform(lstm_pred)[0][0]
    gru_price = scaler.inverse_transform(gru_pred)[0][0]
    cnn_price = scaler.inverse_transform(cnn_pred)[0][0]

    # Append the predictions to respective lists
    lstm_predictions.append(lstm_price)
    gru_predictions.append(gru_price)
    cnn_predictions.append(cnn_price)

    # Update the input sequence with the latest prediction for the next day's prediction
    new_input = np.array([lstm_pred[0, 0]])  # Use LSTM prediction for updating input
    input_sequence = np.append(input_sequence[0, 1:], new_input).reshape(1, sequence_length, 1)

# Create a list of dates for the predictions
prediction_dates = [start_date + timedelta(days=i) for i in range(prediction_days)]

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame({
    'Date': prediction_dates,
    'LSTM_Predicted_Price': lstm_predictions,
    'GRU_Predicted_Price': gru_predictions,
    'CNN_Predicted_Price': cnn_predictions
})

# Save predictions to CSV
predictions_df.to_csv('data/predicted_prices.csv', index=False)
print("Predictions saved to 'data/predicted_prices.csv'")

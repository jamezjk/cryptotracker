import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(data, feature='close'):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[feature]])
    return scaled_data, scaler

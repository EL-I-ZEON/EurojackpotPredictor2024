import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Data preparation
data = {
    'Date': [
        '2024-07-16', '2024-07-12', '2024-07-09', '2024-07-05', '2024-07-02', '2024-06-28',
        '2024-06-25', '2024-06-21', '2024-06-18', '2024-06-14'
    ],
    'Numbers': [
        [8, 22, 27 , 36, 43, 5, 8],
        [2, 14, 30, 32, 34, 3, 4],
        [5, 14, 25, 26, 44, 8, 10],
        [4, 11, 16, 25, 32, 1, 11],
        [10, 29, 30, 32, 40, 6, 12],
        [1, 8, 30, 43, 45, 10, 12],
        [8, 14, 25, 31, 45, 3, 12],
        [2, 22, 24, 30, 40, 5, 6],
        [4, 10, 23, 24, 45, 7, 8],
        [10, 21, 27, 42, 46, 2, 6]
    ]
}

# Convert to numpy array
dates = np.array(data['Date'], dtype='datetime64')
numbers = np.array(data['Numbers'])


# Function to create sequences with dates and numbers
def create_sequences_with_dates(dates, numbers, n_steps):
    X, y = [], []
    for i in range(n_steps, len(numbers)):
        seq_dates = (dates[i - n_steps:i] - dates[i - n_steps]).astype('timedelta64[D]').astype(int)
        seq_numbers = numbers[i - n_steps:i].flatten()
        X.append(np.concatenate((seq_dates, seq_numbers)))
        y.append(numbers[i])
    return np.array(X), np.array(y)


# Create sequences with dates
X, y = create_sequences_with_dates(dates, numbers, 3)

# Scale the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Reshape for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))


# Build LSTM model
def build_model(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(50))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    return model


# Function to train the model and make predictions
def train_and_predict(X, y, last_3_combined_scaled, scaler_y):
    n_steps, n_features = X.shape[1], X.shape[2]
    model = build_model(n_steps, n_features)
    model.fit(X, y, epochs=100, batch_size=1, verbose=2)

    prediction = model.predict(last_3_combined_scaled)
    prediction = scaler_y.inverse_transform(prediction).flatten()

    # Round the predicted numbers to the nearest integers
    predicted_numbers = np.round(prediction).astype(int)
    return predicted_numbers


# Prepare the input for prediction
last_3_dates = (dates[-3:] - dates[-3]).astype('timedelta64[D]').astype(int)
last_3_numbers = numbers[-3:].flatten()
last_3_combined = np.concatenate((last_3_dates, last_3_numbers)).reshape(1, -1)

# Scale the combined last 3 inputs using the scaler for X
last_3_combined_scaled = scaler_X.transform(last_3_combined).reshape((1, 1, last_3_combined.shape[1]))

# Make 5 different predictions
n_predictions = 5
predictions = []
for _ in range(n_predictions):
    predicted_numbers = train_and_predict(X, y, last_3_combined_scaled, scaler_y)
    predictions.append(predicted_numbers)

# Print all 5 predictions
for i, predicted_numbers in enumerate(predictions):
    print(f"Prediction {i + 1}:")
    print("Numbers: ", predicted_numbers[:5])
    print("Euro Numbers: ", predicted_numbers[5:])
    print()
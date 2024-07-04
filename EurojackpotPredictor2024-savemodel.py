import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.models import load_model

# Data preparation
data = {
    'Date': [
        '2024-07-02', '2024-06-28', '2024-06-25', '2024-06-21', '2024-06-18', '2024-06-14', '2024-06-11',
        '2024-06-07', '2024-06-04', '2024-05-31', '2024-05-28', '2024-05-24', '2024-05-21', '2024-05-17',
        '2024-05-14', '2024-05-10', '2024-05-07', '2024-05-03', '2024-04-30', '2024-04-26', '2024-04-23',
        '2024-04-19', '2024-04-16', '2024-04-12', '2024-04-09', '2024-04-05', '2024-04-02', '2024-03-29',
        '2024-03-26', '2024-03-22', '2024-03-19', '2024-03-15', '2024-03-12', '2024-03-08', '2024-03-05',
        '2024-03-01', '2024-02-27', '2024-02-23', '2024-02-20', '2024-02-16', '2024-02-13', '2024-02-09',
        '2024-02-06', '2024-02-02', '2024-01-30', '2024-01-26', '2024-01-23', '2024-01-19', '2024-01-16',
        '2024-01-12', '2024-01-09', '2024-01-05', '2024-01-02'
    ],
    'Numbers': [
        [10, 29, 30, 32, 40, 6, 12], [1, 8, 30, 43, 45, 10, 12], [8, 14, 25, 31, 45, 3, 12], [2, 22, 24, 30, 40, 5, 6],
        [4, 10, 23, 24, 45, 7, 8], [10, 21, 27, 42, 46, 2, 6], [4, 12, 16, 29, 31, 1, 9], [8, 15, 29, 37, 45, 5, 10],
        [1, 3, 24, 43, 49, 2, 4], [4, 23, 34, 39, 45, 6, 7], [13, 26, 27, 35, 46, 3, 4], [2, 3, 4, 21, 45, 6, 12],
        [7, 23, 31, 33, 38, 10, 11], [1, 2, 29, 36, 48, 1, 11], [19, 22, 23, 24, 27, 1, 6], [28, 31, 39, 45, 49, 8, 11],
        [3, 11, 32, 33, 35, 3, 11], [9, 17, 36, 40, 45, 5, 7], [4, 20, 33, 37, 45, 8, 9], [3, 18, 23, 29, 47, 5, 12],
        [2, 3, 6, 15, 35, 1, 3], [8, 14, 21, 34, 36, 1, 2], [35, 36, 37, 41, 48, 1, 12], [1, 34, 39, 47, 49, 1, 12],
        [1, 7, 21, 27, 43, 1, 3], [5, 8, 16, 30, 37, 1, 10], [14, 17, 29, 32, 45, 1, 2], [7, 11, 30, 31, 39, 5, 10],
        [12, 15, 17, 30, 32, 1, 6], [5, 17, 36, 37, 50, 3, 7], [1, 20, 28, 32, 49, 3, 10], [16, 20, 25, 30, 49, 3, 10],
        [2, 8, 11, 16, 20, 4, 10], [2, 11, 17, 23, 49, 4, 12], [2, 20, 30, 31, 40, 8, 12], [13, 26, 30, 34, 41, 3, 7],
        [15, 17, 30, 38, 49, 1, 11], [10, 19, 22, 37, 41, 2, 6], [1, 3, 11, 15, 30, 4, 10], [7, 11, 17, 18, 34, 3, 5],
        [7, 20, 22, 45, 48, 10, 12], [4, 10, 11, 20, 22, 7, 10], [16, 19, 20, 26, 44, 1, 4],
        [13, 17, 21, 30, 39, 8, 11],
        [10, 12, 15, 46, 48, 9, 11], [18, 23, 35, 37, 41, 6, 7], [9, 18, 20, 32, 39, 5, 8], [10, 12, 18, 33, 47, 7, 10],
        [6, 19, 32, 39, 42, 4, 9], [3, 31, 34, 43, 45, 6, 9], [9, 12, 26, 41, 47, 7, 10], [11, 30, 32, 45, 47, 3, 10],
        [19, 26, 36, 48, 49, 10, 11]
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df['Numbers'] = df['Numbers'].apply(lambda x: [int(n) for n in x])

# Sort by Date to ensure chronological order
df = df.sort_values('Date').reset_index(drop=True)

# Prepare features and targets
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Create sequences with different lengths
X_2, y_2 = create_sequences(df['Numbers'].tolist(), 2)
X_3, y_3 = create_sequences(df['Numbers'].tolist(), 3)
X_4, y_4 = create_sequences(df['Numbers'].tolist(), 4)
X_5, y_5 = create_sequences(df['Numbers'].tolist(), 5)
X_6, y_6 = create_sequences(df['Numbers'].tolist(), 6)

# Scale the data
scaler = MinMaxScaler()

# Flatten and scale the sequences
def scale_sequences(X, y):
    n_samples, n_steps, n_features = X.shape
    X_flattened = X.reshape(n_samples * n_steps, n_features)
    y_flattened = y.reshape(len(y), n_features)

    X_scaled_flattened = scaler.fit_transform(X_flattened)
    y_scaled = scaler.transform(y_flattened)

    X_scaled = X_scaled_flattened.reshape(n_samples, n_steps, n_features)

    return X_scaled, y_scaled

X_2_scaled, y_2_scaled = scale_sequences(X_2, y_2)
X_3_scaled, y_3_scaled = scale_sequences(X_3, y_3)
X_4_scaled, y_4_scaled = scale_sequences(X_4, y_4)
X_5_scaled, y_5_scaled = scale_sequences(X_5, y_5)
X_6_scaled, y_6_scaled = scale_sequences(X_6, y_6)

# Build LSTM model
def build_model(n_steps):
    model = Sequential()
    model.add(Input(shape=(n_steps, 7)))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the models
model_2 = build_model(2)
model_2.fit(X_2_scaled, y_2_scaled, epochs=100, batch_size=1, verbose=1)

model_3 = build_model(3)
model_3.fit(X_3_scaled, y_3_scaled, epochs=100, batch_size=1, verbose=1)

model_4 = build_model(4)
model_4.fit(X_4_scaled, y_4_scaled, epochs=100, batch_size=1, verbose=1)

model_5 = build_model(5)
model_5.fit(X_5_scaled, y_5_scaled, epochs=100, batch_size=1, verbose=1)

model_6 = build_model(6)
model_6.fit(X_6_scaled, y_6_scaled, epochs=100, batch_size=1, verbose=1)

# Save models
model_2.save('model_2.h5')
model_3.save('model_3.h5')
model_4.save('model_4.h5')
model_5.save('model_5.h5')
model_6.save('model_6.h5')

# Make predictions
def make_predictions(model, last_draws, scaler, n_steps):
    last_draws = np.array(last_draws).reshape(n_steps, 7)
    last_draws_scaled = scaler.transform(last_draws)
    last_draws_scaled = last_draws_scaled.reshape((1, n_steps, 7))
    prediction = model.predict(last_draws_scaled)
    return scaler.inverse_transform(prediction).flatten()

last_2_draws = df['Numbers'][-2:].tolist()
last_3_draws = df['Numbers'][-3:].tolist()
last_4_draws = df['Numbers'][-4:].tolist()
last_5_draws = df['Numbers'][-5:].tolist()
last_6_draws = df['Numbers'][-6:].tolist()

prediction_2 = make_predictions(model_2, last_2_draws, scaler, 2)
prediction_3 = make_predictions(model_3, last_3_draws, scaler, 3)
prediction_4 = make_predictions(model_4, last_4_draws, scaler, 4)
prediction_5 = make_predictions(model_5, last_5_draws, scaler, 5)
prediction_6 = make_predictions(model_6, last_6_draws, scaler, 6)

print("Predicted winning numbers for the next draw based on last 2 draws:")
print("Numbers: ", prediction_2[:5].astype(int))
print("Euro Numbers: ", prediction_2[5:].astype(int))

print("Predicted winning numbers for the next draw based on last 3 draws:")
print("Numbers: ", prediction_3[:5].astype(int))
print("Euro Numbers: ", prediction_3[5:].astype(int))

print("Predicted winning numbers for the next draw based on last 4 draws:")
print("Numbers: ", prediction_4[:5].astype(int))
print("Euro Numbers: ", prediction_4[5:].astype(int))

print("Predicted winning numbers for the next draw based on last 5 draws:")
print("Numbers: ", prediction_5[:5].astype(int))
print("Euro Numbers: ", prediction_5[5:].astype(int))

print("Predicted winning numbers for the next draw based on last 6 draws:")
print("Numbers: ", prediction_6[:5].astype(int))
print("Euro Numbers: ", prediction_6[5:].astype(int))

# Save the scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')
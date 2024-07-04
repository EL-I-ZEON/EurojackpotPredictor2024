import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load models
model_2 = load_model('model_2.h5')
model_3 = load_model('model_3.h5')
model_4 = load_model('model_4.h5')
model_5 = load_model('model_5.h5')
model_6 = load_model('model_6.h5')

# Load scaler
scaler = joblib.load('scaler.pkl')

# Make predictions using the loaded models
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
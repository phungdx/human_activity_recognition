import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras

body_swing_df = pd.read_csv('BODYSWING.txt')
hand_swing_df = pd.read_csv('HANDSWING.txt')

# print(body_swing_df.shape)
no_of_timesteps = 10

X = []
y = []

dataset = body_swing_df.iloc[:, 1:].values
n_samples = len(dataset)
for i in range(no_of_timesteps, n_samples):
    X.append(dataset[i - no_of_timesteps:i,:])
    y.append(1)

dataset = hand_swing_df.iloc[:, 1:].values
for i in range(no_of_timesteps, n_samples):
    X.append(dataset[i - no_of_timesteps:i,:])
    y.append(0)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs=10)
model.save('model.h5')
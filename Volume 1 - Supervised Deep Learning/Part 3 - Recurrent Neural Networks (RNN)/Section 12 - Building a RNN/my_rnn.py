# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importing the Keras libraries and packages
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, model_from_json


# Importing the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# Shape info https://keras.io/layers/recurrent/
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True,
                   input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the second LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the third LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
# Optimizer info https://keras.io/optimizers/
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 3 - Making the predictions and visualizing the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat(
    (dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# OPTIONAL: Save the model for future uses
# serialize model to JSON
model_json = regressor.to_json()
with open("rnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("rnn_model.h5")
print("Saved model to disk")

## OPTIONAL: Loading the saved model from disk
#json_file = open('rnn_model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#new_regressor = model_from_json(loaded_model_json)
## load weights into new model
#new_regressor.load_weights('rnn_model.h5')
#print("Loaded model from disk")
## Compiling the new model for it's previous use
#new_regressor.compile(optimizer='adam', loss='mean_squared_error')

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue',
         label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

## Evaluation the RNN with RMSE (Root Mean Squared Error)
#import math
#from sklearn.metrics import mean_squared_error
#rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
#print(f"Relative Error: {800/rmse}")
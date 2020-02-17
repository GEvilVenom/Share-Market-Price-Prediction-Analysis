# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('dataset_train.csv')

#Changing The NAN value
dataset_train = dataset_train.interpolate()

training_set = dataset_train.iloc[:, 4:5].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Defining Time Steps
time_step = 60

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(time_step, len(training_set)):
    X_train.append(training_set_scaled[i-time_step:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fitting the RNN to the Training set
epoch_value = 500
history = regressor.fit(X_train, y_train, epochs = epoch_value, batch_size = 40)
print(history.history['loss'])
loss = history.history['loss']

#Graph_Analysis_Of_Loss
plt.figure(figsize=(8,4))
plt.plot(loss[1:epoch_value], color = 'red', label = 'Loss')
plt.title('Loss_Graph_Analysis')
plt.xlabel('Time')
plt.ylabel('Loss Percentage')
plt.show()



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('dataset_test.csv')

#Changing The NAN value
dataset_test = dataset_test.interpolate()
real_stock_price = dataset_test.iloc[:, 4:5].values
real_stock_demo_price = real_stock_price

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(time_step,(time_step+len(real_stock_price)+1)):
    X_test.append(inputs[i-time_step:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price_temp = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_temp)

for j in range(1,10):
    
    last_predicted_data =  predicted_stock_price_temp[len(real_stock_price)+j-1]
    inputs = np.append(inputs,last_predicted_data)
    inputs = inputs.reshape(-1,1)
    ##inputs = sc.transform(inputs)
    X_test = []
    for i in range(time_step,(time_step+len(real_stock_price)+1+j)):
        X_test.append(inputs[i-time_step:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price_temp = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price_temp)
    
    
#Error Calculation
error = []
error_percentage = []
for i in range (0,len(real_stock_price)):
    if(real_stock_price[i]<=predicted_stock_price[i+1]):
        error_value = predicted_stock_price[i+1] - real_stock_price[i]
    else:
        error_value = real_stock_price[i] - predicted_stock_price[i+1]
        
    error_percentage_value = error_value/real_stock_price[i]
    error.append(error_value)
    error_percentage.append(error_percentage_value)
    
error_rescaled = sc.fit_transform(error)

#Error_Graph_analysis
plt.figure(figsize=(8,4))
plt.plot(error_percentage, color = 'red', label = 'Error')
plt.title('Error_Graph_Analysis')
plt.xlabel('Time')
plt.ylabel('Error Percentage')
plt.show()

#for Indexing
acceptable_prediction = 0
rejected_prediction = 0

for i in range (0,len(error_rescaled)):
    if(error_rescaled[i]<=0.5):
        acceptable_prediction = acceptable_prediction + 1
    else:
        rejected_prediction = rejected_prediction + 1
    
print("percentage of Acceptable_Prediction : ",(acceptable_prediction/len(error_rescaled))*100)

###############################################################################
value = 0
for i in range(0,len(real_stock_price)):
    value = value + real_stock_price[i] - predicted_stock_price[i+1]
value = value/len(real_stock_price)
predicted_stock_price_dummy = []
for i in range(0,len(predicted_stock_price)):
    predicted_stock_price_dummy.append(predicted_stock_price[i]+value)

# Visualising the results
plt.figure(figsize=(8,4))
plt.plot(real_stock_price, color = 'red', label = 'Real Dataset Stock Price')
plt.plot(predicted_stock_price[1:], color = 'blue', label = 'Predicted Dataset Stock Price')
plt.title('Dataset Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Dataset Stock Price')
plt.legend()
plt.show()

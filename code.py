#importing necessary libraries
import numpy as np
import pandas
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import os
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
start_time = time.time()


#converting data
def prepare_data(dataset, features_count):
	X, y =[],[]
	for i in range(len(dataset)):
		end_ix = i + features_count
		if end_ix > len(dataset)-1:
			break
		seq_x, seq_y = dataset[i:end_ix], dataset[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

dataframe = pandas.read_csv('Data\monthly_tourist_arrivals_india_2002-2018.csv', usecols=[0], engine='python')
dataset = dataframe.values

# setting parameters
Previous_data_length = len(dataset) #original data shown in the graph , here:whole data
forecast_size=12 # how many months ahead we want to forecast
Iterations=1 #no. of training iterations
step_count = 12 #no.of values to feed the model to train on at a time (dont change)

X, y = prepare_data(dataset, step_count)

features_count = 1
X = X.reshape((X.shape[0], X.shape[1], features_count))
final_batch_size=3 # can set it to 12 for faster training but reduces accuracy
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(step_count, features_count)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, batch_size=final_batch_size,epochs=Iterations,validation_split=0.0833, verbose=1) #0.0833 is 11/12 means train on 11 and test on 1 .

X_input = np.asarray(dataset[len(dataset)-step_count-forecast_size:len(dataset)-forecast_size,:]).astype('float32')
T_input=list(X_input) 
Output=[]
i=0

#take some(forecast_size-1) of last values , predict 1 values , insert that values as input ,predict new value , and continue  

while(i<forecast_size):
    
    if(len(T_input)>step_count):
        X_input=np.asarray(T_input[1:]).astype('float32')
        print("{} month input {}".format(i,X_input))
        X_input = X_input.reshape((1, step_count, features_count))
        prediction_value = model.predict(X_input, verbose=0) #predict the value
        print("{} month output {}".format(i,prediction_value))
        T_input.append(prediction_value[0][0])#store predicted output to work as input for next iteration
        T_input=T_input[1:] #remove the first value as we inserted new value
        Output.append(prediction_value[0][0])#store all the predicted value
        i=i+1
        print(prediction_value[0][0])
    else:
        X_input = X_input.reshape((1, step_count, features_count))
        prediction_value = model.predict(X_input, verbose=0)
        print(prediction_value[0])
        T_input.append(prediction_value[0][0])
        Output.append(prediction_value[0][0])
        i=i+1
        print(prediction_value[0][0])
    

print(Output)

pevious_data=np.arange(0,Previous_data_length) # the original data to be shown on graph
new_data=np.arange(Previous_data_length-forecast_size,Previous_data_length) # the predicted data

rmse = math.sqrt(mean_squared_error(dataset[len(dataset)-forecast_size:len(dataset),:], Output))
print('Test RMSE: %.4f' % rmse)

end_time = time.time()
print("Total time: ", (end_time - start_time )/60, "Minutes or" , (end_time - start_time)%60, "seconds")

plt.plot(pevious_data,dataset[len(dataset)-Previous_data_length:len(dataset),:]) #plot original data in graph 
plt.plot(new_data,Output)# plot predicted data in graph
plt.show() #show the plots

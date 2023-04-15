#math/plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#AI libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



def main():   
    ###TRAINING FUNCTION START###
    apple_training_complete = pd.read_csv(r'AAPL_training.csv')
    apple_training_processed = apple_training_complete.iloc[:, 1:2].values   
    scaler = MinMaxScaler(feature_range = (0, 1))

    apple_training_scaled = scaler.fit_transform(apple_training_processed)
    
    features_set = []
    labels = []
    for i in range(60, len(apple_training_scaled)):
        features_set.append(apple_training_scaled[i-60:i, 0])
        labels.append(apple_training_scaled[i, 0])

    
    features_set, labels = np.array(features_set), np.array(labels)

    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    #make the model
    model = Sequential()
    
    #add an lstm layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    #then add a dropout layer to prevent overfitting to training data
    model.add(Dropout(0.2))

    #add another couple lstm/droupout layers
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    #now add a Dense layer that condenses the inputs into a single value output
    model.add(Dense(units = 1))

    #now compile the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
 
    #now train the model using our features and label
    model.fit(features_set, labels, epochs = 100, batch_size = 32)
    ###TRAINING FUNCTION END###
    
    #now test the model
    apple_testing_complete = pd.read_csv(r'AAPL_testing.csv')
    apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values

    apple_total = pd.concat((apple_training_complete['Open'], apple_testing_complete['Open']), axis=0)
 
    test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values

    #reshape the test data like we did with the training data
    test_inputs = test_inputs.reshape(-1,1)
    test_inputs = scaler.transform(test_inputs)

    test_features = []
    for i in range(60, 80):
        test_features.append(test_inputs[i-60:i, 0])
    
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
  
    predictions = model.predict(test_features)

    predictions = scaler.inverse_transform(predictions)

    plt.figure(figsize=(10,6))
    plt.plot(apple_testing_processed, color='blue', label='Actual Apple Stock Price')
    plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Apple Stock Price')
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
        main()


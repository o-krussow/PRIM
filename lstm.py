#math/plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#AI libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, History

#AI optimizing Libraries
#from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim
#from hyperas.distributions import choice, uniform

#File IO
from file_handler import File

class Model:
    def __init__(self, data_file_name, time_span = 60):
        self._time_span = time_span
        self._batch_size = 32
        self._epochs = 100
        #open the data file
        self._df = pd.read_csv(data_file_name)
        self._df_train = self._df.iloc[:, :].values
        self._df_train_unscaled = self._df.iloc[:, :].values

        #scale the data to [0, 1]
        self._scaler = MinMaxScaler(feature_range = (0, 1))
        self._df_train = self._scaler.fit_transform(self._df_train)
        self._create_model()

    def _get_data(self):
        return self._df_train

    def _create_model(self):
        self._features_set = []
        self._labels = []
        for i in range(self._time_span, len(self._df_train)):
            self._features_set.append(self._df_train[i-self._time_span:i, 0])
            self._labels.append(self._df_train[i, 0])

        self._features_set, self._labels = np.array(self._features_set), np.array(self._labels)
        self._features_set = np.reshape(self._features_set, (self._features_set.shape[0], self._features_set.shape[1], 1))
 
        #make the model
        self._model = Sequential()

        #add an lstm layer
        self._model.add(LSTM(units=50, return_sequences=True, input_shape=(self._features_set.shape[1], 1)))
        #then add a dropout layer to prevent overfitting to training data
        self._model.add(Dropout(0.2))
        
        #add another couple lstm/droupout layers
        self._model.add(LSTM(units=50, return_sequences=True))
        self._model.add(Dropout(0.2))

        self._model.add(LSTM(units=50, return_sequences=True))
        self._model.add(Dropout(0.2))

        self._model.add(LSTM(units=50))
        self._model.add(Dropout(0.2))

        #now add a Dense layer that condenses the inputs into a single value output
        self._model.add(Dense(units = 1))
     
        #now compile the model
        self._model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    def train(self, epochs = 100, batch_size = 32):
        self._batch_size = batch_size
        #now train the model using our features and label
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                           min_delta=0.001)
        history = History()
        self._model.fit(self._features_set, self._labels, 
                        epochs = epochs, batch_size = self._batch_size,
                        verbose = 1,
                        validation_split = 0.2, 
                        callbacks=[es, history])

    def predict(self, futurecast):
        predictions = []
        for i in range(0, futurecast):
            predictions.append(self._predict_next_timestep())

        self._df_train_unscaled = self._df.iloc[:, :].values

        return predictions

    def _predict_next_timestep(self):
        test_inputs = self._df_train_unscaled[-self._time_span:]
        test_inputs = test_inputs.reshape(-1,1)
        test_inputs = self._scaler.transform(test_inputs)

        test_features = []
        test_features.append(test_inputs[0:self._time_span, 0])

        test_features = np.array(test_features)   
        test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

        prediction = self._model.predict(test_features, verbose = 0)
        prediction = self._scaler.inverse_transform(prediction)
        self._df_train_unscaled = np.append(self._df_train_unscaled, prediction)
        
        return prediction[0][0]

    """
    def hyperfit(self):
        best_run = optim.minimize(model = self._create_model,
                                  data = self._get_data,
                                  algo = tpe.suggest,
                                  max_evals = 5,
                                  trials = Trials())
        self._df_train = self._get_data()
        print('hi')
    """

def main():
    """
    ###TRAINING FUNCTION START###
    apple_training_complete = pd.read_csv(r'AAPL_training.csv')
    apple_training_processed = apple_training_complete.iloc[:, 1:2].values   
    scaler = MinMaxScaler(feature_range = (0, 1))

    apple_training_scaled = scaler.fit_transform(apple_training_processed)
    
    features_set = []
    self._self._labels = []
    for i in range(60, len(apple_training_scaled)):
        features_set.append(apple_training_scaled[i-60:i, 0])
        self._self._labels.append(apple_training_scaled[i, 0])

    
    features_set, self._self._labels = np.array(features_set), np.array(labels)
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
    model.fit(features_set, self._self._labels, epochs = 100, batch_size = 32)
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
    """
    
    model = Model("BASMX-10y-1d.csv")
    model.train(1, 30)
    predictions = model.predict(20)

    plt.figure(figsize=(10,6))
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()



    

if __name__ == "__main__":
        main()


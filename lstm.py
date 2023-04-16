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
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.backend import get_session

#File IO
from file_handler import File
import os

import time

class Model:
    def __init__(self, data_file_name, time_span = 60):
        #data_type = "Stock Prices", x_axis_label = "Date"):
        #self._x_axis_label = x_axis_label
        #self._data_type = data_type
        self._time_span = time_span
        self._batch_size = 32
        self._epochs = 100
        #open the data file
        self._df = pd.read_csv(data_file_name)
        self._df_train = self._df.iloc[:, :].values
        self._df_train_unscaled = self._df.iloc[:, :].values
        
        #scale the data to [0, 1]
        self._scaler = MinMaxScaler(feature_range = (0, 1))
        self._df_train = self._scaler.fit_transform(self._df_train_unscaled)

        self._create_model()

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
        self._model.add(LSTM(60, return_sequences=True,
                             input_shape = (self._features_set.shape[1], 1)))
        #then add a dropout layer to prevent overfitting to training data
        self._model.add(Dropout(0.4))
        
        #add another couple lstm/dropout layers
        self._model.add(LSTM(30, dropout = 0.0))
        self._model.add(Dropout(0.4))

        #now add a Dense layer that condenses the inputs into a single value output
        self._model.add(Dense(20, activation = 'relu'))
        self._model.add(Dense(1, activation = 'sigmoid'))

        #now compile the model
        self._model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        #save initial blank weights
        self._model.save_weights('model_blank.h5')
    
    def train(self, epochs = 100, batch_size = 32, save = False):

        self._batch_size = batch_size
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_path,
                                                    save_weights_only = True,
                                                    monitor = 'val_loss',
                                                    mode = 'min',
                                                    save_best_only = True)

        #now train the model using our features and label
        es = EarlyStopping(monitor = 'val_loss', 
                           patience = 20,   #Number of epochs with no improvement after which training will be stopped.
                           min_delta = .00005,  #Minimum change in the monitored quantity to qualify as improvement
                           start_from_epoch = 20,
                           verbose = 1,
                           restore_best_weights= True)

        self.history = History()
        result = self._model.fit(self._features_set, self._labels, 
                        epochs = epochs, batch_size = self._batch_size,
                        verbose = 1, 
                        validation_split = 0.2, 
                        callbacks=[model_checkpoint_callback, self.history, es])
        validation_loss = np.amin(result.history['val_loss'])
        self._model.load_weights(checkpoint_path)
        
        if save:
            self._model.save_weights("model.h5")

        return validation_loss

    def load_weights(self, path_to_weights):
        self._model.load_weights(path_to_weights)
        
    def hyperfit(self, epochs = [50, 100], batch_sizes = [7, 14, 32], save = False):
        #lower fitness value is better
        #(best fitting model, fitness value of best model)
        best_model = (None,  0)
        for epoch in epochs:
            for batch_size in batch_sizes:
                fitness = self.train(epoch, batch_size)
                if fitness < best_model[1] or best_model[0] == None:
                    best_model = (self._model, fitness)
                self._reset_weights()

        self._model = best_model[0]
        if save:
            self._model.save_weights("model.h5")

        #return to model_manager for graphs
        return (self.history.history['loss'], self.history.history['val_loss'])

    def _reset_weights(self):
        self._model.load_weights('model_blank.h5')

    def predict(self, futurecast):
        # generate the multi-step forecasts
        data_future = []

        feature_set_pred = [self._df_train[-self._time_span:]]  # last observed input sequence
        price_pred = [[self._df_train[-1]]]                         # last observed target value
        feature_set_pred = np.array(feature_set_pred)
        price_pred = np.array(price_pred)
        for i in range(futurecast):

            # feed the last forecast back to the model as an input
            feature_set_pred = np.append(feature_set_pred[:, 1:, :], price_pred.reshape(1, 1, 1), axis = 1)

            # generate the next forecast
            price_pred = self._model.predict(feature_set_pred)

            # save the forecast
            data_future.append(price_pred.flatten()[0])

        # transform the forecasts back to the original scale
        data_future = np.array(data_future).reshape(-1, 1)
        data_future = self._scaler.inverse_transform(data_future)

        data_future = data_future.flatten().tolist()
        print(data_future)
        return data_future

    def test_30_days(self):
        # generate the multi-step forecasts
        data_future = []

        feature_set_pred = [self._df_train[-self._time_span:]]  # last observed input sequence
        price_pred = [[self._df_train[-1]]]                         # last observed target value
        feature_set_pred = np.array(feature_set_pred)
        price_pred = np.array(price_pred)
        for i in range(30):

            # feed the last forecast back to the model as an input
            feature_set_pred = np.append(feature_set_pred[:, 1:, :], price_pred.reshape(1, 1, 1), axis = 1)

            # generate the next forecast
            price_pred = self._model.predict(feature_set_pred)

            # save the forecast
            data_future.append(price_pred.flatten()[0])

        # transform the forecasts back to the original scale
        data_future = np.array(data_future).reshape(-1, 1)
        data_future = self._scaler.inverse_transform(data_future)

        return data_future, self._df_train_unscaled[-30:]

    def _test(self):
        size = 300
        x = np.zeros((size, 60, 1))
        y = np.zeros((size,))
        for i in range(size):
            x[i] = self._df_train[i: 60+i]
            y[i] = self._df_train[i]
        
        temp = self._model.predict(x)

        plt.figure(figsize=(10,6))
        plt.plot(y, color='blue', label=f"Actual price")
        plt.plot(temp, color='red', label=f'Predicted price')
        plt.title(f'Stock Price Prediction')
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
        
def main():
    
    start_time = time.time()
    
    model = Model("FSKAX-10y-1d.csv")
    
    #model.train(50, 14, save = True)
    model.hyperfit(save = True)
    model.load_weights('model.h5')
    
    predictions = model.predict(25)

    for i in range(100):
        predictions.insert(0, None)

    data = model._df_train_unscaled[-100:]

    #model.test()
    
    plt.figure(figsize=(10,6))
    plt.plot(data, color='blue', label=f"Actual price")
    plt.plot(predictions, color='red', label=f'Predicted price')
    plt.title(f'Stock Price Prediction')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


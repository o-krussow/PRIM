import pickle
import json
import lstm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

class model_manager:

    def __init__(self):
        self.models = {} #Not going to make getters and setters, just easier to access this dictionary directly I think
        self._ticker_prices_csv_path = "./ticker_prices/"
        #self._graph_dir = "./graphs/"
        self._pickled_models_path = "./pickled_models/" #adjust if you put pickled files in sub directory
        
        try:
            with open("trained-models.json") as f:
                self.tickers = json.load(f)


            #depickle models for tickers
            for ticker in self.tickers: #Assuming pickled file name is just the ticker
                self.models[ticker] = pickle.load(open(self._pickled_models_path+ticker, "rb"))

        except FileNotFoundError:
            #If file does not exist, we do nothing
            pass



    def graphs(self, model_output, ticker_name):
        
        plt.figure(figsize=(10,6))
        plt.plot(model_output[0], color='blue', label=f'Actual {ticker_name} Stock Price')
        plt.plot(model_output[1], color='red', label=f'Predicted {ticker_name} Stock Price')
        plt.title(f'{ticker_name} Stock Price Prediction')
        plt.xlabel('Time (Days)')
        plt.ylabel(f'{ticker_name} Stock Price')
        plt.legend()
        plt.savefig("./static/"+ticker_name+"_test_output.jpg") # PUT ALL IMAGES IN STATIC FOR FLASK
        

        plt.figure(figsize=(10,6))
        plt.plot(model_output[2], color='blue', label='Training Loss')
        plt.plot(model_output[3], color='red', label='Testing Loss')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("./static/"+ticker_name+"_error_output.jpg") # PUT ALL IMAGES IN STATIC FOR FLASK


    def add_model(self, ticker_name):

        self._get_stock_data(ticker_name)
        #now there will be a ticker_name.csv file in self._ticker_prices_csv_path

        model = lstm.Model(self._ticker_prices_csv_path+ticker_name+".csv") #opening that csv
        model_output = model.hyperfit() #not sure if this is the right prediction data, given that it is the test data and not fit but I'm not sure.

        self.graphs(model_output, ticker_name)

        self.models[ticker_name] = model #add trained model to dictionary, the "" is just a placeholder

        #pickle the new trained model
        self._pickle_model(ticker_name)


        self._commit_json() #this just updates our list of tickers that we have a model for

    def get_ticker_prediction(self, ticker_name, periods):
        model_keys = list(self.models.keys())
        if ticker_name not in model_keys: #if ticker has already been trained for
            self.add_model(ticker_name)
        
        requested_model = self.models[ticker_name]

        return requested_model.predict(periods)



    def _get_stock_data(self, ticker_name, period="10y", interval="1d"):
        stock_object = yf.Ticker(ticker_name)

        hist = stock_object.history(period=period, interval=interval)

        stock_data_list = hist["Close"].values.tolist()

        #i don't want to have to do it this way but whatever
        with open(self._ticker_prices_csv_path+ticker_name+".csv", "w+") as f:
            for entry in stock_data_list:
                f.write(str(entry)+"\n")



    def _pickle_model(self, ticker): #takes an instance of the model class and pickles it into the pickle directory
        model = self.models[ticker]

        with open(self._pickled_models_path+ticker, "wb") as f:
            pickle.dump(model, f) #so we can access later

    def _commit_json(self):
        with open("trained-models.json", "w") as f:
            model_tickers = list(self.models.keys())
            json.dump(model_tickers, f, ensure_ascii=False)


def main():
    mm = model_manager()


    print(mm.get_ticker_prediction("BASMX", 10))
    print(mm.ticker_times["BASMX"])

if __name__ == "__main__":
    main()
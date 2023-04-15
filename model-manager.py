import pickle
import json
import lstm
import matplotlib.pyplot as plt

class model_manager:

    def __init__(self):
        self.models = {} #Not going to make getters and setters, just easier to access this dictionary directly I think

        ticker_files_path = "./pickled_models/" #adjust if you put pickled files in sub directory

        try:
            with open("trained-models.json") as f:
                tickers = json.load()

            #depickle models for tickers
            for ticker in tickers: #Assuming pickled file name is just the ticker
                self.models[ticker] = pickle.load(open(ticker_files_path+ticker, "rb"))

        except FileNotFoundError:
            with open("trained-models.json") as f:
                json.dump(f, []) #just make empty json if it doesn't exist already, this doesn't really NEED to happen because commit_json will do it anyways but it's fine


    def tune_model(self, csv_path): #I named this totally arbitrarily, I don't really know how we want to do this part
        #run inference/somehow call predict function in pierce's thing however he decides to make it work

        model = lstm.Model(csv_path)
        model_output = model.hyperfit()
        self.graphs(model_output)

        pass

    def graphs(self, model_output):
        plt.figure(figsize=(10,6))
        plt.plot(model_output[0], color='blue', label='Actual Stock Price')
        plt.plot(model_output[1] , color='red', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig("test_output.jpg")

        plt.figure()
        plt.plot(model_output[2])
        plt.plot(model_output[3])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("error_output.jpg")


    def add_model(self, ticker_name):

        #need to:
        #instantiate new model (model takes csv path and name)
        #CSV will be uploaded from client to server, not sure yet where it'll pop up here yet. This function might have to be put inside the flask code, along with a lot of this class.
        #train the model

        self.models[ticker_name] = "" #add trained model to dictionary, the "" is just a placeholder

        #pickle the new trained model
        pickle_model(ticker_name)

        commit_json() #this just updates our list of tickers that we have a model for

    def pickle_model(self, ticker):
        model = self.models[ticker]

        with open(ticker_files_path+ticker, "wb") as f:
            pickle.dump(model, f) #so we can access later

    def commit_json(self):
        with open("trained-models.json", "w") as f:
            json.dump(self.models.keys(), f, ensure_ascii=False)



import pickle
import json

class model_manager:


    def __init__(self):
        self.models = {} #Not going to make getters and setters, just easier to access this dictionary directly I think
        try:
            with open("trained-models.json") as f:
                tickers = json.load()

            #depickle models for tickers
            for ticker in tickers: #Assuming pickled file name is just the ticker
                self.models[ticker] = pickle.load(open(ticker, "r"))

        except FileNotFoundError:
            with open("trained-models.json") as f:
                json.dump(f, []) #just make empty json if it doesn't exist already, this doesn't really NEED to happen because commit_json will do it anyways but it's fine


    def access_model(self, ticker_name):
        #run inference/somehow call predict function in pierce's thing however he decides to make it work
        continue

    def add_model(self, ticker_name):

        #need to:
        #instantiate new model (model takes csv path and name)
        #train the model

        self.models[ticker_name] = "" #add trained model to dictionary

        #pickle the new trained model
        pickle_model(ticker_name)

        commit_json() #this just updates our list of tickers that we have a model for

    def pickle_model(self, ticker):
        model = self.models[ticker]

        with open(ticker, "w+") as f:
            pickle.dump(model, f)

    def commit_json(self):
        with open("trained-models.json", "w") as f:
            json.dump(self.models.keys(), f, ensure_ascii=False)



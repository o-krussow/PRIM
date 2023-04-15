import pickle
import json

class model_manager:
    models = {} #Not going to make getters and setters, just easier to access this dictionary directly I think


    def __init__():
        try:
            with open("trained-models.json") as f:
                tickers = json.load()

            #depickle models for tickers
            for ticker in tickers: #Assuming pickled file name is just the ticker
                models[ticker] = pickle.load(open(ticker, "r"))

        except FileNotFoundError:
            with open("trained-models.json") as f:
                json.dump(f, [])


    def access_model(ticker_name):
        continue
        #run inference/somehow call predict function in pierce's thing however he decides to make it work

    def add_model(ticker_name):

        #need to:
        #instantiate new model (model takes csv path and name)
        #train the model

        models[ticker_name] = "" #add trained model to dictionary

        #pickle the new trained model
        pickle_model(ticker_name)

        commit_json() #this just updates our list of tickers that we have a model for

    def pickle_model(ticker):
        model = models[ticker]

        with open(ticker, "w+") as f:
            pickle.dump(model, f)

    def commit_json():
        with open("trained-models.json", "w") as f:
            json.dump(models.keys(), f, ensure_ascii=False)



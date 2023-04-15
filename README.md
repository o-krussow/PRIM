
## ARCHITECTURE

Web server imports model manager

Model manager imports model

When user wants to train a model for a ticker, web server recieves post request and tells model manager to train a new model

 - Model manager training:
   - First check if ticker has a model, and make sure it isn't expired.
	 - If there is no existing model or existing model is expired, we fetches price data from yfinance and continue
	 - Passes price data to model class, then tells it to start training
	 - Model will return data from training, like loss data, which we will probably want to send back to the user in the form of a graph. So this data will be sent to model manager.
	 - Once model is finished training, model manager pickles the model and saves it to local filesystem
	 - Adds model to the model dictionary, {key: model}
 - Web server needs to send data back to the user
   - So a component creates the graph files / results in a certain folder
	 - Then the front end needs to http get these files, so it can be shown to the user in their browser






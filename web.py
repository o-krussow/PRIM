from model_manager import model_manager
from flask import Flask, render_template, redirect
from flask import request
import json
import io
from matplotlib.figure import Figure

app = Flask(__name__)
results = []

@app.route("/")
def main():

	return render_template("index.html", name = results)

def predict():
	modelManager = model_manager();
	output = request.get_json() # These next two lines get the values from the Json request made in index.js
	result = json.loads(output)
	stockName = result["stock"] # Get stock name input
	amountDays = int(result["time"]) # Get amount of days input
	print(modelManager.get_ticker_prediction(result["stock"], int(result["time"]))) # Call get_ticker_prediction in model_manager.py to get predict values and make graph
	testGraphName = "/static/" + stockName + "_test_output.jpg"; # Get name of image (all follow same format, NEED TO BE IN STATIC FOLDER)
	errorGraphName = "/static/" + stockName + "_error_output.jpg";
	return testGraphName;
	return render_template("test.html", testGraph=testGraphName, errorGraph = errorGraphName)

@app.route('/predict', methods=['POST']) # JSON REQUEST GOES HERE FIRST, ENTERS predict() AND RETURNS NAME OF TEST GRAPH IMAGE
def renderPredict():
	return predict() # Returns name of test graph image (GO TO index.js LINE 66)

if __name__ == "__main__":
	app.run(debug=True, port=6958) # IF PORT ALREADY IN USE, JUST DECREASE PORT NUMBER

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
	output = request.get_json()
	result = json.loads(output)
	stockName = result["stock"]
	amountDays = int(result["time"])
	print(modelManager.get_ticker_prediction(result["stock"], int(result["time"])))
	testGraphName = "/static/" + stockName + "_test_output.jpg";
	errorGraphName = "/static/" + stockName + "_error_output.jpg";
	return testGraphName;
	return render_template("test.html", testGraph=testGraphName, errorGraph = errorGraphName)

@app.route('/predict', methods=['POST'])
def renderPredict():
	return predict()

if __name__ == "__main__":
	app.run(debug=True, port=6959)

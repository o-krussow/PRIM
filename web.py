from model_manager import model_manager
from flask import Flask, render_template
from flask import request
import json

app = Flask(__name__)

@app.route("/")
def main():

	return render_template("index.html")

@app.route('/test', methods=['POST'])
def test():
	modelManager = model_manager();
	output = request.get_json()
	result = json.loads(output)
	print(modelManager.get_ticker_prediction(result["stock"], int(result["time"])))
	return result



if __name__ == "__main__":
	app.run(debug=True, port=6968)

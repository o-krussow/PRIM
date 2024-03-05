
# PRIM

## What it does
PRIM uses an Long Short Term Model (LSTM) neural network to predict a stock's price in "x" amount of days. Users interact with PRIM through a web app, and input a stock ticker and prediction day and are returned with a graph of the predicted price.

## How we built it
The front end is built using HTML, CSS, JavaScript, with JSon and Flask constituting the middle end connecting to a backend of entirely Python. We used the Yahoo Finance API to scrape financial data from stocks when given a ticker, and we used Keras Library to build, train, test, and deploy the LSTM. Finally, we used Pickle to store the most accurate models for future use.

## Challenges we ran into
The biggest challenge we faced was with our LSTM. It took most of our second day of development to produce even remotely accurate predictions due to some bugs and difficult data structures. On the frontend, it was difficult to load the produced graphs from the backend. Finally, a major challenge that we didn't consider during the hackathon was the actual logic of our prediction - that predicting farther than 1 day out compounded error and the predictions would never be viable outside of 1 day ahead.

## Accomplishments that we're proud of
We are proud of building and training our first ever Machine Learning model, and for successfully learning and using Flask and JSON for the first time.

## What we learned
We learned almost everything needed for this project during the hackathon. Everything about LSTMs: how they work, how to build, train, test, and implement; everything with the middle end: Flask and JSON connecting Python and HTML; and a lot of general project management skills: GIT, branches, and merging.

## What's next for PRIM
We plan to redesign PRIM to focus solely on 1 day forecasting, using Interactive Brokers API for more data. We also plan on revamping the data it is given, potentially letting it compare many stocks and choose the best ones to invest in.

## Built With
css, flask, html, javascript, json, keras, matplotlib, numpy, pandas, pickle, python, yahoo-finance

![PRIM Prediction (60 day time set)](https://user-images.githubusercontent.com/123573986/232597844-9b79442d-e93b-46ef-adaf-878b70b5211a.png)

PRIM is accurate when predicting only 1 day out using 60 days of actual data

![PRIM Prediction (60 to 30)](https://user-images.githubusercontent.com/123573986/232597846-328ea2cc-8382-44c8-9361-2447dddac413.png)

When predicting farther out, PRIM loses viability due to compunding error of using its own predictions as data.

## To Test the LSTM:
Download the repository and run lstm.py. You'll have to install relevant dependencies.
Python will let you know what dependencies are left when you try to run lstm.py.
Installing dependencies is done through pip.
(Note: the command to install sklearn is pip install scikit-learn)


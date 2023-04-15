#!/usr/local/bin/python

import yfinance as yf
import json

index_funds = ["FSKAX", "SWTSX", "VTSAX", "VFTAX", "FZROX", "FITLX", "BASMX"]

for stock_name in index_funds:

    speriod = "60d"
    sinterval = "1d"

    stock = yf.Ticker(stock_name)

    hist = stock.history(period=speriod, interval=sinterval)

    listized = hist["Close"].values.tolist()

    print("Number of points:", len(listized))

    with open(stock_name+"-"+speriod+"-"+sinterval+".json", "w") as f:
        json.dump(listized, f, ensure_ascii=False)



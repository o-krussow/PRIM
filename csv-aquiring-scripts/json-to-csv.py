import json
import sys

filename = sys.argv[1]

with open(filename, "r") as file:
    prices = json.load(file)

with open(filename.replace("json", "csv"), "w+") as file:
    for price in prices:
        file.write(str(price)+"\n")

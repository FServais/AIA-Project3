from pandas import read_csv

import numpy as np

data = read_csv("train_data.csv", index_col="TRIP_ID")

# Replace non-numeric values

# CALL_TYPE
data.loc[data["CALL_TYPE"] == 'A', "CALL_TYPE"] = 0
data.loc[data["CALL_TYPE"] == 'B', "CALL_TYPE"] = 1
data.loc[data["CALL_TYPE"] == 'C', "CALL_TYPE"] = 2

# DAY_TYPE
data.loc[data["DAY_TYPE"] == 'A', "DAY_TYPE"] = 0
data.loc[data["DAY_TYPE"] == 'B', "DAY_TYPE"] = 1
data.loc[data["DAY_TYPE"] == 'C', "DAY_TYPE"] = 2

# MISSING_DATA
data.loc[data["MISSING_DATA"] == True, "MISSING_DATA"] = 0
data.loc[data["MISSING_DATA"] == False, "MISSING_DATA"] = 1

# print(data.head(5))
# print(data.describe())

# Extract 'y'
rides = data['POLYLINE'].values

X, y = rides[:-1], rides[-1]


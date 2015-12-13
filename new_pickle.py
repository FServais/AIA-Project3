import datetime
import numpy as np
import pandas as pd


def from_time_to_day_period(row):
    st = datetime.datetime.fromtimestamp(row["TIMESTAMP"]).strftime('%H')
    hour = int(st)

    return hour

if __name__ == "__main__":
    # Data Loading
    # TRAIN_SET_SIZE = 1500000
    print("Reading values...")
    data_read = pd.read_csv('train_data.csv', index_col="TRIP_ID")
    # data = data_read.sample(frac=1)
    data = data_read

    # Extract 'y' and long and lat
    rides = data['POLYLINE'].values
    rides = list(map(eval, rides))

    data['POLYLINE'] = rides

    data = data[data["POLYLINE"].map(len) > 2]

    # CALL_TYPE
    data.loc[data["CALL_TYPE"] == 'A', "CALL_TYPE"] = 0
    data.loc[data["CALL_TYPE"] == 'B', "CALL_TYPE"] = 1
    data.loc[data["CALL_TYPE"] == 'C', "CALL_TYPE"] = 2

    # DAY_TYPE
    data.loc[data["DAY_TYPE"] == 'A', "DAY_TYPE"] = 0
    data.loc[data["DAY_TYPE"] == 'B', "DAY_TYPE"] = 1
    data.loc[data["DAY_TYPE"] == 'C', "DAY_TYPE"] = 2

    # MISSING_DATA
    data.loc[data["MISSING_DATA"] == True, "MISSING_DATA"] = 1
    data.loc[data["MISSING_DATA"] == False, "MISSING_DATA"] = 0

    data["TIMESTAMP"] = data.apply(from_time_to_day_period, axis=1)

    # Delete all the datas which have missing data in their paths
    data = data[data["MISSING_DATA"] != 1]

    data = data[["TIMESTAMP", "POLYLINE", "TAXI_ID"]]

    data.to_pickle('new_data_pickle.pkl')
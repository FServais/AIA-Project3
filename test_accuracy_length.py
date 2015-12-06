# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import cross_validation, grid_search
from sklearn.neighbors import KNeighborsRegressor
from itertools import repeat
from math import floor
import datetime

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def print_submission(trip_id, result, name):
    n_line, n_columns = result.shape
    with open(name + '.txt', 'w') as f:
        f.write('"TRIP_ID","LATITUDE","LONGITUDE"\n')
        for i in range(n_line):
            line = '"{}",{},{}\n'.format(trip_id[i], result[i,0], result[i,1])
            f.write(line)


def get_last_coordinate(l):
    """
    [[a,b], [c,d], [e,f]] -> [e,f]
    :param l:
    :return:
    """
    return l[-1]


def repeat_list(l, times_repeated):
    return [x for item in l for x in repeat(item, times_repeated)]


def expand_list(l, final_size):
    l_size = len(l)
    if final_size <= l_size or l_size <= 0:
        return l

    repeated_list = []
    repeated_list.extend(l)

    mult_each_el = int(floor(final_size/l_size))
    remaining = final_size - (mult_each_el * l_size)

    t1 = []
    t2 = []
    if remaining > 0:
        t1 = repeat_list(l[:remaining], mult_each_el+1)

    t2 = repeat_list(l[remaining:], mult_each_el)

    return t1 + t2


def from_time_to_day_period(row):
    st = datetime.datetime.fromtimestamp(row["TIMESTAMP"]).strftime('%H')
    hour = int(st)

    return hour

if __name__ == "__main__":
    predictors = ["CALL_TYPE", "DAY_TYPE", "TAXI_ID", "TIMESTAMP"]

    # Data Loading
    TRAIN_SET_SIZE = 1000000
    data = pd.read_csv('train_data.csv', index_col="TRIP_ID", nrows=TRAIN_SET_SIZE)
    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

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
    data.loc[data["MISSING_DATA"] == True, "MISSING_DATA"] = 1
    data.loc[data["MISSING_DATA"] == False, "MISSING_DATA"] = 0

    data["TIMESTAMP"] = data.apply(from_time_to_day_period, axis=1)

    data["ORIGIN_CALL"] = data["ORIGIN_CALL"].fillna(round(data["ORIGIN_CALL"].mean()))

    data["ORIGIN_STAND"] = data["ORIGIN_STAND"].fillna(round(data["ORIGIN_STAND"].mean()))

    # Delete all the datas which have missing data in their paths
    data = data[data["MISSING_DATA"] != 1]

    train_size = int(0.7 * TRAIN_SET_SIZE)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Extract 'y' and long and lat
    rides = train_data['POLYLINE'].values
    rides = list(map(eval, rides))

    origins = []
    length_rides = []

    # For each ride
    for i in range(len(rides)):
        dist = len(rides[i])

        if dist <= 1:
            continue

        origins.append([rides[i][0][0], rides[i][0][1]] + [data[f].iloc[i] for f in predictors])
        length_rides.append(dist)

    knn_len = DecisionTreeRegressor()
    knn_len.fit(origins, length_rides)

    rides_test = test_data['POLYLINE'].values
    rides_test = list(map(eval, rides_test))

    accuracies = np.zeros(len(rides_test))

    # For each path
    for i in range(len(rides_test)):
        dist = len(rides_test[i])

        if dist < 1:
            continue
        # Predict the length of the path
        c = np.array([rides_test[i][0][0], rides_test[i][0][1]] + [test_data[f].iloc[i] for f in predictors])
        predicted_len = knn_len.predict(c.reshape(1,-1))
        predicted_len = predicted_len[0]

        accuracies[i] = abs(dist-predicted_len)


# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import cross_validation, grid_search
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
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
@    TRAIN_SET_SIZE = 1500000
    data = pd.read_csv('train_data.csv', index_col="TRIP_ID", nrows=TRAIN_SET_SIZE)
    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

    # Replace non-numeric values
    #
    # # CALL_TYPE
    # data.loc[data["CALL_TYPE"] == 'A', "CALL_TYPE"] = 0
    # data.loc[data["CALL_TYPE"] == 'B', "CALL_TYPE"] = 1
    # data.loc[data["CALL_TYPE"] == 'C', "CALL_TYPE"] = 2
    #
    # # DAY_TYPE
    # data.loc[data["DAY_TYPE"] == 'A', "DAY_TYPE"] = 0
    # data.loc[data["DAY_TYPE"] == 'B', "DAY_TYPE"] = 1
    # data.loc[data["DAY_TYPE"] == 'C', "DAY_TYPE"] = 2
    #
    # # MISSING_DATA
    # data.loc[data["MISSING_DATA"] == True, "MISSING_DATA"] = 1
    # data.loc[data["MISSING_DATA"] == False, "MISSING_DATA"] = 0
    #
    # data["TIMESTAMP"] = data.apply(from_time_to_day_period, axis=1)
    #
    # data["ORIGIN_CALL"] = data["ORIGIN_CALL"].fillna(round(data["ORIGIN_CALL"].mean()))
    #
    # data["ORIGIN_STAND"] = data["ORIGIN_STAND"].fillna(round(data["ORIGIN_STAND"].mean()))

    # Delete all the datas which have missing data in their paths
    data = data[data["MISSING_DATA"] != 1]

    # Extract 'y' and long and lat
    rides = data['POLYLINE'].values
    rides = list(map(eval, rides))

    # Separate inputs and outputs by number of features
    X_25 = []
    X_50 = []
    X_100 = []
    X_plus = []

    y_25 = []
    y_50 = []
    y_100 = []
    y_plus = []

    longest_ride_length = 0

    # X,y used to predict the path
    origins = []
    length_rides = []

    # For each ride
    for i in range(len(rides)):
        dist = len(rides[i])

        if dist <= 1:
            continue

        # Find length of the longest ride
        if dist > longest_ride_length:
            longest_ride_length = dist

        # Split LAT/LONG
        long = []
        lat = []

        for c in range(dist-1):
            long.append(rides[i][c][0])
            lat.append(rides[i][c][1])

        # Add the data to the corresponding <X,y>
        if dist <= 25:
            X_25.append(expand_list(long, 25) + expand_list(lat, 25))
            y_25.append([rides[i][-1][0], rides[i][-1][1]])
        elif dist <= 50:
            X_50.append(expand_list(long, 50) + expand_list(lat, 50))
            y_50.append([rides[i][-1][0], rides[i][-1][1]])
        elif dist <= 100:
            X_100.append(expand_list(long, 100) + expand_list(lat, 100))
            y_100.append([rides[i][-1][0], rides[i][-1][1]])
        else:
            X_plus.append([long, lat])
            y_plus.append([rides[i][-1][0], rides[i][-1][1]])

        # origins.append([rides[i][0][0], rides[i][0][1]] + [data[f].iloc[i] for f in predictors])
        origins.append([rides[i][0][0], rides[i][0][1]])
        length_rides.append(dist)

    # Correct X_plus (i.e. expand long and lat)
    for i in range(len(X_plus)):
        X_plus[i] = expand_list(X_plus[i][0], longest_ride_length) + expand_list(X_plus[i][1], longest_ride_length)

    # Training
    knn_25 = RandomForestRegressor(n_jobs=-1)
    knn_50 = RandomForestRegressor(n_jobs=-1)
    knn_100 = RandomForestRegressor(n_jobs=-1)
    knn_plus = RandomForestRegressor(n_jobs=-1)

    knn_25.fit(X_25, y_25)
    knn_50.fit(X_50, y_50)
    knn_100.fit(X_100, y_100)
    knn_plus.fit(X_plus, y_plus)

    knn_len = KNeighborsRegressor(n_neighbors=51)
    knn_len.fit(origins, length_rides)

    # Test Set Loading
    test = pd.read_csv('test.csv', index_col="TRIP_ID")
    n_trip_test, _ = test.shape
    print('Shape of test data: {}'.format(test.shape))
    trip_id = list(test.index)

    # Replace non-numeric values
    #
    # # CALL_TYPE
    # test.loc[test["CALL_TYPE"] == 'A', "CALL_TYPE"] = 0
    # test.loc[test["CALL_TYPE"] == 'B', "CALL_TYPE"] = 1
    # test.loc[test["CALL_TYPE"] == 'C', "CALL_TYPE"] = 2
    #
    # # DAY_TYPE
    # test.loc[test["DAY_TYPE"] == 'A', "DAY_TYPE"] = 0
    # test.loc[test["DAY_TYPE"] == 'B', "DAY_TYPE"] = 1
    # test.loc[test["DAY_TYPE"] == 'C', "DAY_TYPE"] = 2
    #
    # # MISSING_DATA
    # test.loc[test["MISSING_DATA"] == True, "MISSING_DATA"] = 1
    # test.loc[test["MISSING_DATA"] == False, "MISSING_DATA"] = 0
    #
    # test["TIMESTAMP"] = test.apply(from_time_to_day_period, axis=1)
    #
    # test["ORIGIN_CALL"] = test["ORIGIN_CALL"].fillna(round(test["ORIGIN_CALL"].mean()))
    #
    # test["ORIGIN_STAND"] = test["ORIGIN_STAND"].fillna(round(test["ORIGIN_STAND"].mean()))

    rides_test = test['POLYLINE'].values
    rides_test = list(map(eval, rides_test))

    X_test = []
    X_long_test = []
    X_lat_test = []

    y_predict = []
    # For each path
    for i in range(len(rides_test)):
        dist = len(rides_test[i])

        if dist < 1:
            continue

        X_lat_test_tmp = []
        X_long_test_tmp = []

        # Predict the length of the path
        # c = np.array([rides_test[i][0][0], rides_test[i][0][1]] + [test[f].iloc[i] for f in predictors])
        c = np.array([rides_test[i][0][0], rides_test[i][0][1]])
        predicted_len = knn_len.predict(c.reshape(1,-1))
        predicted_len = predicted_len[0]

        # In case the predicted length is smaller than what's provided
        if predicted_len < dist:
            predicted_len = dist

        # For each coordinate
        for coord in rides_test[i]:
            X_long_test_tmp.append(coord[0])
            X_lat_test_tmp.append(coord[1])

        # Predict the point
        if predicted_len <= 25:
            X_to_predict = np.array(expand_list(X_long_test_tmp, 25) + expand_list(X_lat_test_tmp, 25))
            prediction = knn_25.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])
        elif predicted_len <= 50:
            X_to_predict = np.array(expand_list(X_long_test_tmp, 50) + expand_list(X_lat_test_tmp, 50))
            prediction = knn_50.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])
        elif predicted_len <= 100:
            X_to_predict = np.array(expand_list(X_long_test_tmp, 100) + expand_list(X_lat_test_tmp, 100))
            prediction = knn_100.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])
        else:
            X_to_predict = np.array(expand_list(X_long_test_tmp, longest_ride_length) + expand_list(X_lat_test_tmp, longest_ride_length))
            prediction = knn_plus.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])

    result = np.zeros((n_trip_test, 2))

    for i in range(len(y_predict)):
        result[i, 0] = y_predict[i][1]
        result[i, 1] = y_predict[i][0]

    # Write submission
    print_submission(trip_id=trip_id, result=result, name="test_mult_sampleSubmission_generated")

    print("End of prediction")

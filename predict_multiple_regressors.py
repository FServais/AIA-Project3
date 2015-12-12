# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import cross_validation, grid_search, preprocessing
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelBinarizer
from itertools import repeat
from math import floor
import datetime

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sknn.mlp import Regressor, Layer


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
    predictors = ["CALL_TYPE", "TIMESTAMP"]

    # Data Loading
    # TRAIN_SET_SIZE = 1500000
    print("Reading values...")
    data_read = pd.read_csv('train_data.csv', index_col="TRIP_ID", nrows=150000)
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

    lb = preprocessing.LabelBinarizer()
    lb.fit(data["TAXI_ID"].values)
    taxiId = lb.transform(data["TAXI_ID"].values)

    for i in range(len(taxiId[0])):
        data[str("A"+str(i))] = taxiId[:,i]

    # predictors = ["CALL_TYPE", "TIMESTAMP", "DIRECTION"] + ["A{}".format(i) for i in range(len(taxiId[0]))]


    predictors = ["TIMESTAMP"] + [str('A'+str(i)) for i in range(len(taxiId[0]))]
    # predictors = ["TIMESTAMP", "TAXI_ID"]


    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

    # Extract 'y' and long and lat
    rides = data['POLYLINE'].values

    # Separate inputs and outputs by number of features
    X_25 = []
    X_50 = []
    X_100 = []
    X_plus = []

    y_25 = []
    y_50 = []
    y_100 = []
    y_plus = []

    longest_ride_length = 4000

    # X,y used to predict the path
    origins = []
    length_rides = []

    print("Preprocessing")

    # For each ride
    for i in range(len(rides)):
        if i%1000 == 0:
            print("Process ride {}".format(i))

        dist = len(rides[i])

        if dist < 2:
            continue

        # Find length of the longest ride
        # if dist > longest_ride_length:
        #     longest_ride_length = dist

        # Split LAT/LONG
        long = []
        lat = []

        for c in range(dist-1):
            long.append(rides[i][c][0])
            lat.append(rides[i][c][1])

        preds = [data[f].iloc[i] for f in predictors]

        # Add the data to the corresponding <X,y>
        if dist <= 25:
            X_25.append(expand_list(long, 25) + expand_list(lat, 25) + preds)
            y_25.append([rides[i][-1][0], rides[i][-1][1]])
        elif dist <= 50:
            X_50.append(expand_list(long, 50) + expand_list(lat, 50) + preds)
            y_50.append([rides[i][-1][0], rides[i][-1][1]])
        elif dist <= 100:
            X_100.append(expand_list(long, 100) + expand_list(lat, 100) + preds)
            y_100.append([rides[i][-1][0], rides[i][-1][1]])
        else:
            X_plus.append(expand_list(long, longest_ride_length) + expand_list(lat, longest_ride_length) + preds)
            y_plus.append([rides[i][-1][0], rides[i][-1][1]])

        origins.append([rides[i][0][0], rides[i][0][1]] + preds)
        length_rides.append(dist)

    print("Training")

    # Training
    knn_25 = DecisionTreeRegressor()
    knn_50 = DecisionTreeRegressor()
    knn_100 = DecisionTreeRegressor()
    knn_plus = DecisionTreeRegressor()

    knn_25.fit(np.array(X_25), np.array(y_25))
    knn_50.fit(np.array(X_50), np.array(y_50))
    knn_100.fit(np.array(X_100), np.array(y_100))
    knn_plus.fit(np.array(X_plus), np.array(y_plus))

    knn_len = DecisionTreeRegressor()
    knn_len.fit(origins, length_rides)

    print("End of training")

    # Test Set Loading
    test = pd.read_csv('test.csv', index_col="TRIP_ID")
    n_trip_test, _ = test.shape
    print('Shape of test data: {}'.format(test.shape))
    trip_id = list(test.index)

    # Replace non-numeric values

    # CALL_TYPE
    test.loc[test["CALL_TYPE"] == 'A', "CALL_TYPE"] = 0
    test.loc[test["CALL_TYPE"] == 'B', "CALL_TYPE"] = 1
    test.loc[test["CALL_TYPE"] == 'C', "CALL_TYPE"] = 2

    # DAY_TYPE
    test.loc[test["DAY_TYPE"] == 'A', "DAY_TYPE"] = 0
    test.loc[test["DAY_TYPE"] == 'B', "DAY_TYPE"] = 1
    test.loc[test["DAY_TYPE"] == 'C', "DAY_TYPE"] = 2

    # MISSING_DATA
    test.loc[test["MISSING_DATA"] == True, "MISSING_DATA"] = 1
    test.loc[test["MISSING_DATA"] == False, "MISSING_DATA"] = 0

    test["TIMESTAMP"] = test.apply(from_time_to_day_period, axis=1)

    test["ORIGIN_CALL"] = test["ORIGIN_CALL"].fillna(round(test["ORIGIN_CALL"].mean()))

    test["ORIGIN_STAND"] = test["ORIGIN_STAND"].fillna(round(test["ORIGIN_STAND"].mean()))

    rides_test = test['POLYLINE'].values
    rides_test = list(map(eval, rides_test))

    lb = preprocessing.LabelBinarizer()
    lb.fit(data["TAXI_ID"].values)
    taxiId = lb.transform(test["TAXI_ID"].values)

    for i in range(len(taxiId[0])):
        test[str("A"+str(i))] = taxiId[:,i]

    print("Test preprocessing")

    X_test = []
    X_long_test = []
    X_lat_test = []

    y_predict = []
    # For each path
    for i in range(len(rides_test)):
        if i%1000 == 0:
            print("Process ride {}".format(i))

        dist = len(rides_test[i])

        if dist < 1:
            continue

        X_lat_test_tmp = []
        X_long_test_tmp = []

        preds = [test[f].iloc[i] for f in predictors]

        # Predict the length of the path
        c = np.array([rides_test[i][0][0], rides_test[i][0][1]] + preds)
        # c = np.array([rides_test[i][0][0], rides_test[i][0][1]])
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
            X_to_predict = np.array(expand_list(X_long_test_tmp, 25) + expand_list(X_lat_test_tmp, 25) + preds)
            prediction = knn_25.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])
        elif predicted_len <= 50:
            X_to_predict = np.array(expand_list(X_long_test_tmp, 50) + expand_list(X_lat_test_tmp, 50) + preds)
            prediction = knn_50.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])
        elif predicted_len <= 100:
            X_to_predict = np.array(expand_list(X_long_test_tmp, 100) + expand_list(X_lat_test_tmp, 100) + preds)
            prediction = knn_100.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])
        else:
            X_to_predict = np.array(expand_list(X_long_test_tmp, longest_ride_length) + expand_list(X_lat_test_tmp, longest_ride_length) + preds)
            prediction = knn_plus.predict(X_to_predict.reshape(1,-1))
            y_predict.append(prediction[0])

    print("Writing")
    result = np.zeros((n_trip_test, 2))

    for i in range(len(y_predict)):
        result[i, 0] = y_predict[i][1]
        result[i, 1] = y_predict[i][0]

    # Write submission
    print_submission(trip_id=trip_id, result=result, name="taxiid_mult_sampleSubmission_generated")

    print("End of prediction")

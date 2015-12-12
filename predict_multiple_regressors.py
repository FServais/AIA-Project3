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
    data_read = pd.read_pickle('dir_data_pickle.pkl')
    data = data_read.sample(frac=1)
    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

    # Extract 'y' and long and lat
    rides = data['POLYLINE'].values

    # Separate inputs and outputs by number of features
    X_25_long = []
    X_25_lat = []
    X_50_long = []
    X_50_lat = []
    X_100_long = []
    X_100_lat = []
    X_plus_long = []
    X_plus_lat = []

    y_25_long = []
    y_25_lat = []
    y_50_long = []
    y_50_lat = []
    y_100_long = []
    y_100_lat = []
    y_plus_long = []
    y_plus_lat = []

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
            X_25_long.append(expand_list(long, 25) + preds)
            X_25_lat.append(expand_list(lat, 25) + preds)
            y_25_long.append([rides[i][-1][0]])
            y_25_lat.append([rides[i][-1][1]])
        elif dist <= 50:
            X_50_long.append(expand_list(long, 50) + preds)
            X_50_lat.append(expand_list(lat, 50) + preds)
            y_50_long.append([rides[i][-1][0]])
            y_50_lat.append([rides[i][-1][1]])
        elif dist <= 100:
            X_100_long.append(expand_list(long, 100) + preds)
            X_100_lat.append(expand_list(lat, 100) + preds)
            y_100_long.append([rides[i][-1][0]])
            y_100_lat.append([rides[i][-1][1]])
        else:
            X_plus_long.append(expand_list(long, longest_ride_length) + preds)
            X_plus_lat.append(expand_list(lat, longest_ride_length) + preds)
            y_plus_long.append([rides[i][-1][0]])
            y_plus_lat.append([rides[i][-1][1]])

        origins.append([rides[i][0][0], rides[i][0][1]] + preds)
        length_rides.append(dist)

    print("Training")

    # Training
    knn_25_long = DecisionTreeRegressor()
    knn_25_lat = DecisionTreeRegressor()
    knn_50_long = DecisionTreeRegressor()
    knn_50_lat = DecisionTreeRegressor()
    knn_100_long = DecisionTreeRegressor()
    knn_100_lat = DecisionTreeRegressor()
    knn_plus_long = DecisionTreeRegressor()
    knn_plus_lat = DecisionTreeRegressor()

    knn_25_long.fit(np.array(X_25_long), np.array(y_25_long))
    knn_25_lat.fit(np.array(X_25_lat), np.array(y_25_lat))
    knn_50_long.fit(np.array(X_50_long), np.array(y_50_long))
    knn_50_lat.fit(np.array(X_50_lat), np.array(y_50_lat))
    knn_100_long.fit(np.array(X_100_long), np.array(y_100_long))
    knn_100_lat.fit(np.array(X_100_lat), np.array(y_100_lat))
    knn_plus_long.fit(np.array(X_plus_long), np.array(y_plus_long))
    knn_plus_lat.fit(np.array(X_plus_lat), np.array(y_plus_lat))

    knn_len = DecisionTreeRegressor()
    knn_len.fit(origins, length_rides)

    print("End of training")

    # Test Set Loading
    test = pd.read_pickle('dir_test_pickle.pkl')
    n_trip_test, _ = test.shape
    print('Shape of test data: {}'.format(test.shape))
    trip_id = list(test.index)

    rides_test = test['POLYLINE'].values

    print("Test preprocessing")

    X_test = []
    X_long_test = []
    X_lat_test = []

    y_predict_long = []
    y_predict_lat = []
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
            X_to_predict = np.array(expand_list(X_long_test_tmp, 25) + preds)
            prediction = knn_25_long.predict(X_to_predict.reshape(1,-1))
            y_predict_long.append(prediction[0])

            X_to_predict = np.array(expand_list(X_lat_test_tmp, 25) + preds)
            prediction = knn_25_lat.predict(X_to_predict.reshape(1,-1))
            y_predict_lat.append(prediction[0])
        elif predicted_len <= 50:
            X_to_predict = np.array(expand_list(X_long_test_tmp, 50) + preds)
            prediction = knn_50_long.predict(X_to_predict.reshape(1,-1))
            y_predict_long.append(prediction[0])

            X_to_predict = np.array(expand_list(X_lat_test_tmp, 50) + preds)
            prediction = knn_50_lat.predict(X_to_predict.reshape(1,-1))
            y_predict_lat.append(prediction[0])
        elif predicted_len <= 100:
            X_to_predict = np.array(expand_list(X_long_test_tmp, 100) + preds)
            prediction = knn_100_long.predict(X_to_predict.reshape(1,-1))
            y_predict_long.append(prediction[0])

            X_to_predict = np.array(expand_list(X_lat_test_tmp, 100) + preds)
            prediction = knn_100_lat.predict(X_to_predict.reshape(1,-1))
            y_predict_lat.append(prediction[0])
        else:
            X_to_predict = np.array(expand_list(X_long_test_tmp, longest_ride_length) + preds)
            prediction = knn_plus_long.predict(X_to_predict.reshape(1,-1))
            y_predict_long.append(prediction[0])

            X_to_predict = np.array(expand_list(X_lat_test_tmp, longest_ride_length) + preds)
            prediction = knn_plus_lat.predict(X_to_predict.reshape(1,-1))
            y_predict_lat.append(prediction[0])

    print("Writing")
    result = np.zeros((n_trip_test, 2))

    for i in range(len(y_predict_long)):
        result[i, 0] = y_predict_lat[i]
        result[i, 1] = y_predict_long[i]

    # Write submission
    print_submission(trip_id=trip_id, result=result, name="div_mult_sampleSubmission_generated")

    print("End of prediction")

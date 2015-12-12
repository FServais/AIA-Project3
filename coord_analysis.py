import datetime
from itertools import repeat
from math import floor, ceil

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from random import randint


def bias(predicted, y_ls):
    return np.mean(y_ls) - np.mean(predicted)

def var_ls(y_ls):
    return np.var(y_ls)

def mae(predicted, y):
    return np.sum(np.abs(predicted - y))/len(y)

def extract_rides(data):
    rides = data['POLYLINE'].values
    return np.array(list(map(eval, rides)))

def extract_origins_lengths(rides):
    origins = np.zeros((len(rides),2))
    lengths = np.zeros(len(rides))

    for i in range(len(rides)):
        origins[i][0] = rides[i][0][0]
        origins[i][1] = rides[i][0][1]
        lengths[i] = len(rides[i])

    return origins, lengths

def from_time_to_day_period(row):
    st = datetime.datetime.fromtimestamp(row["TIMESTAMP"]).strftime('%H')
    hour = int(st)

    return hour

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

def extract_subpath(row):
    lower = int(ceil(0.1 * len(row)))+1
    upper = len(row)-1

    l = len(row[0 : (randint(lower,upper))])

    # print("{} to {} ({})".format(lower, upper, l))

    return row[0 : (randint(lower,upper))]

if __name__ == "__main__":

    print("Reading values...")
    data_read = pd.read_pickle('dir_data_pickle_500000.pkl')
    data = data_read.sample(frac=0.1)

    data = data[data["POLYLINE"].map(len) > 3]

    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

    kf = KFold(len(data), n_folds=3)
    # max_depths = np.append(np.arange(10, 100, 20), [None])
    # max_depths = np.append(np.append(np.arange(1,10,1), np.arange(10, 100, 20)), [None])
    # n_est = np.append(np.append(np.arange(1,10,1), np.arange(10, 100, 10)), [None])
    # max_depths = [1]
    # n_neighbs = np.append(np.arange(1,20,2), np.arange(25,200,25))


    lb = preprocessing.LabelBinarizer()
    lb.fit(data["TAXI_ID"].values)
    taxiId = lb.transform(data["TAXI_ID"].values)

    for i in range(len(taxiId[0])):
        data["A{}".format(i)] = taxiId[:,i]

    predictors = ["CALL_TYPE", "TIMESTAMP", "DIRECTION"] + ["A{}".format(i) for i in range(len(taxiId[0]))]

    b = []
    v = []
    e = []
    for train_index, test_index in kf:
        X_ls = data.iloc[train_index]

        # rides_ls = extract_rides(X_ls)
        rides_ls = X_ls["POLYLINE"].values
        origins_ls, lengths_ls = extract_origins_lengths(rides_ls)
        X_ls_predict = X_ls.drop("POLYLINE", 1)
        X_ls_predict["org_x"] = origins_ls[:,0]
        X_ls_predict["org_y"] = origins_ls[:,1]

        regr = DecisionTreeRegressor()
        regr.fit(X_ls_predict, lengths_ls)

        y_ls = np.zeros((len(rides_ls), 2))
        for i in range(len(rides_ls)):
            y_ls[i] = rides_ls[i][-1]
            rides_ls[i] = rides_ls[i][:-1]

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

        # For each ride
        for i in range(len(rides_ls)):
            dist = len(rides_ls[i])

            if dist <= 1:
                continue

            # # Find length of the longest ride
            # if dist > longest_ride_length:
            #     longest_ride_length = dist

            # Split LAT/LONG
            long = []
            lat = []

            for c in range(dist-1):
                long.append(rides_ls[i][c][0])
                lat.append(rides_ls[i][c][1])

            preds = [X_ls[f].iloc[i] for f in predictors]

            # Add the data to the corresponding <X,y>
            if dist <= 25:
                X_25_long.append(expand_list(long, 25) + preds)
                X_25_lat.append(expand_list(lat, 25) + preds)
                y_25_long.append([rides_ls[i][-1][0]])
                y_25_lat.append([rides_ls[i][-1][1]])
            elif dist <= 50:
                X_50_long.append(expand_list(long, 50) + preds)
                X_50_lat.append(expand_list(lat, 50) + preds)
                y_50_long.append([rides_ls[i][-1][0]])
                y_50_lat.append([rides_ls[i][-1][1]])
            elif dist <= 100:
                X_100_long.append(expand_list(long, 100) + preds)
                X_100_lat.append(expand_list(lat, 100) + preds)
                y_100_long.append([rides_ls[i][-1][0]])
                y_100_lat.append([rides_ls[i][-1][1]])
            else:
                X_plus_long.append(expand_list(long, longest_ride_length) + preds)
                X_plus_lat.append(expand_list(lat, longest_ride_length) + preds)
                y_plus_long.append([rides_ls[i][-1][0]])
                y_plus_lat.append([rides_ls[i][-1][1]])


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

        # ========= TEST
        X_test = data.iloc[test_index]

        # rides_test = extract_rides(X_test)
        rides_test = X_test["POLYLINE"].values

        origins_test, lengths_test = extract_origins_lengths(rides_test)
        X_test_predict_len = X_test.drop("POLYLINE", 1)
        X_test_predict_len["org_x"] = origins_test[:,0]
        X_test_predict_len["org_y"] = origins_test[:,1]

        y_test = np.zeros((len(rides_test), 2))
        for i in range(len(rides_test)):
            y_test[i] = rides_test[i][-1]
            rides_test[i] = rides_test[i][:-1]

        predict = regr.predict(X_test_predict_len)

        X_test["DESTINATION"] = X_test["POLYLINE"].apply(lambda row: row[-1])

        X_test["POLYLINE"] = X_test["POLYLINE"].apply(extract_subpath)

        y_predict_long = []
        y_predict_lat = []
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
            predicted_len = predict[i]

            # In case the predicted length is smaller than what's provided
            if predicted_len < dist:
                predicted_len = dist

            for c in range(dist-1):
                X_long_test_tmp.append(rides_test[i][c][0])
                X_lat_test_tmp.append(rides_test[i][c][1])

            preds = [X_test[f].iloc[i] for f in predictors]

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

        LONG = X_test["DESTINATION"].apply(lambda row: row[0]).values
        LAT = X_test["DESTINATION"].apply(lambda row: row[1]).values

        # LONG_PREDICT = np.array(list(map(lambda c: c[0], y_predict_long)))
        # LAT_PREDICT = np.array(list(map(lambda c: c[1], y_predict_lat)))
        LONG_PREDICT = y_predict_long
        LAT_PREDICT = y_predict_lat

        b.append(np.mean([bias(LONG_PREDICT, LONG), bias(LAT_PREDICT, LAT)]))
        v.append(np.mean([var_ls(LONG_PREDICT), var_ls(LAT_PREDICT)]))
        e.append(np.mean([mae(LONG_PREDICT, LONG), mae(LAT_PREDICT, LAT)]))

    print("DecisionTree (Reference)")
    print("Bias: ", np.mean(b))
    print("Variance: ", np.mean(v))
    print("Mean absolute error: ", np.mean(e))
    print("\n")



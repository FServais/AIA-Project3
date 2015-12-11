import datetime
from itertools import repeat
from math import floor

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


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

if __name__ == "__main__":

    data_read = pd.read_pickle('dir_data_pickle.pkl')
    data = data_read.sample(frac=0.05)

    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

    kf = KFold(len(data), n_folds=3)
    nns = np.append(np.append(np.arange(1,11,1), np.arange(15, 50, 5)), np.arange(50, 200, 20))

    for nn in nns:
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

            regr = DecisionTreeRegressor(max_depth=10)
            regr.fit(X_ls_predict, lengths_ls)

            y_ls = np.zeros((len(rides_ls), 2))
            for i in range(len(rides_ls)):
                y_ls[i] = rides_ls[i][-1]
                rides_ls[i] = rides_ls[i][:-1]


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

                # Add the data to the corresponding <X,y>
                if dist <= 25:
                    X_25.append(expand_list(long, 25) + expand_list(lat, 25))
                    y_25.append([rides_ls[i][-1][0], rides_ls[i][-1][1]])
                elif dist <= 50:
                    X_50.append(expand_list(long, 50) + expand_list(lat, 50))
                    y_50.append([rides_ls[i][-1][0], rides_ls[i][-1][1]])
                elif dist <= 100:
                    X_100.append(expand_list(long, 100) + expand_list(lat, 100))
                    y_100.append([rides_ls[i][-1][0], rides_ls[i][-1][1]])
                else:
                    X_plus.append([long, lat])
                    y_plus.append([rides_ls[i][-1][0], rides_ls[i][-1][1]])

            # Correct X_plus (i.e. expand long and lat)
            for i in range(len(X_plus)):
                X_plus[i] = expand_list(X_plus[i][0], longest_ride_length) + expand_list(X_plus[i][1], longest_ride_length)

            # Training

            knn_25 = KNeighborsRegressor(n_neighbors=nn)
            knn_50 = KNeighborsRegressor(n_neighbors=nn)
            knn_100 = KNeighborsRegressor(n_neighbors=nn)
            knn_plus = KNeighborsRegressor(n_neighbors=nn)

            knn_25.fit(np.array(X_25), np.array(y_25))
            knn_50.fit(np.array(X_50), np.array(y_50))
            knn_100.fit(np.array(X_100), np.array(y_100))
            knn_plus.fit(np.array(X_plus), np.array(y_plus))

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
                predicted_len = predict[i]

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

            b.append(np.mean([bias(np.array(y_predict[0]), y_ls), bias(np.array(y_predict[1]), y_ls)]))
            v.append(np.mean([var_ls(np.array(y_predict[0])), var_ls(np.array(y_predict[1]))]))
            e.append(np.mean([mae(np.array(y_predict[0]), y_ls), mae(np.array(y_predict[1]), y_ls)]))

        print("KNN with N = {}".format(nn))
        print("Bias: ", np.mean(b))
        print("Variance: ", np.mean(v))
        print("Mean absolute error: ", np.mean(e))
        print("\n")



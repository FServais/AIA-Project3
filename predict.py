#! /usr/bin/env python
# -*- coding: utf-8 -*-

# How to use this function?
# trip_id is the first column of the submission, i.e., the id of the sample
# result is a two-column matrix made of rows of "Latitude"x"Longitude" pairs.
# 		Note that the line "result[i,:]" should correspond to id "trip_id[i]"
# name is the name you want for your submission file
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


# As an example, when you call "toy_script" as a script,
# it will produce the file "sampleSubmission.csv"

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from itertools import repeat
from math import floor

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
    if final_size <= l_size:
        return l

    repeated_list = []
    repeated_list.extend(l)

    mult_each_el = floor(final_size/l_size)
    remaining = final_size - (mult_each_el * l_size)

    t1 = []
    t2 = []
    if remaining > 0:
        t1 = repeat_list(l[:remaining], mult_each_el+1)

    t2 = repeat_list(l[remaining:], mult_each_el)

    return t1 + t2


if __name__ == "__main__":

    # Data Loading
    data = pd.read_csv('test.csv', index_col="TRIP_ID")
    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

    f = 0
    t = 0
    for i in data["MISSING_DATA"]:
        if i == True:
            t += 1
        else:
            f += 1

    print("False: {} ; True: {}".format(f, t))


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
    
    data["ORIGIN_CALL"] = data["ORIGIN_CALL"].fillna(round(data["ORIGIN_CALL"].mean()))
    
    data["ORIGIN_STAND"] = data["ORIGIN_STAND"].fillna(round(data["ORIGIN_STAND"].mean()))

    #print(data.head(6))
    #print(data.describe())
    
    
    # Extract 'y'
    rides = data['POLYLINE'].values
    rides = list(map(eval, rides))

    X = []
    y = []
    for i in range(len(rides)):
        if len(rides[i]) == 0:
            continue
        if len(rides[i]) < 2:
            X.append(rides[i][0])
        else:
            X.append(rides[i][-2])
        y.append(rides[i][-1])

    # X = np.zeros((len(rides), 2)) # Origin, last step
    # y = np.zeros(len(rides))

    # for r in range(len(rides)):
    #     X[i][0] = rides[i][0]
    #     if len(rides[i]) < 2:
    #         X[i][1] = rides[i][0]
    #     else:
    #         X[i][1] = rides[i][-2]
    #
    #     y[i] = rides

    # Test Set Loading
    test = pd.read_csv('test.csv', index_col="TRIP_ID")
    n_trip_test, _ = test.shape
    print('Shape of test data: {}'.format(test.shape))
    trip_id = list(test.index)

    # Extract 'y'
    rides_test = test['POLYLINE'].values
    rides_test = list(map(eval, rides_test))


    X_test = []
    for i in range(len(rides_test)):
        if len(rides_test[i]) < 2:
            X_test.append(rides_test[i][0])
        else:
            X_test.append(rides_test[i][-2])


    # # How to make timestamp nicer
    # clean_timestamp = pd.to_datetime(data["TIMESTAMP"], unit="s")  
    
    # Training
    dtr = KNeighborsRegressor()
    
    #for i in range(len(X)):
    dtr.fit(X, y)

    # Prediction
    y_predict = dtr.predict(X_test)

    result = np.zeros((n_trip_test, 2))

    for i in range(len(y_predict)):
        result[i, 0] = y_predict[i][1]
        result[i, 1] = y_predict[i][0]

    # Write submission
    print_submission(trip_id=trip_id, result=result, name="sampleSubmission_generated")
    
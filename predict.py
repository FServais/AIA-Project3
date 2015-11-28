#! /usr/bin/env python
# -*- coding: utf-8 -*-

# How to use this function?
# trip_id is the first column of the submission, i.e., the id of the sample
# result is a two-column matrix made of rows of "Latitude"x"Longitude" pairs.
# 		Note that the line "result[i,:]" should correspond to id "trip_id[i]"
# name is the name you want for your submission file
from sklearn.linear_model import LinearRegression
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


def get_last_coordinate(l):
    """
    [[a,b], [c,d], [e,f]] -> [e,f]
    :param l:
    :return:
    """
    return l[-1]


if __name__ == "__main__":

    # Data Loading
    data = pd.read_csv('test.csv', index_col="TRIP_ID")
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
    data.loc[data["MISSING_DATA"] == True, "MISSING_DATA"] = 0
    data.loc[data["MISSING_DATA"] == False, "MISSING_DATA"] = 1
    
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
        X.append(rides[i][:-1])
        y.append(rides[i][-1])

    # Test Set Loading
    test = pd.read_csv('test.csv', index_col="TRIP_ID")
    n_trip_test, _ = test.shape
    print('Shape of test data: {}'.format(test.shape))
    trip_id = list(test.index)

    # Extract 'y'
    rides_test = test['POLYLINE'].values

    X_test = list(map(eval, rides_test[:-1]))
    

    # # How to make timestamp nicer
    # clean_timestamp = pd.to_datetime(data["TIMESTAMP"], unit="s")  
    
    # Training
    dtr = DecisionTreeRegressor()
    
    #for i in range(len(X)):
    dtr.fit(X, y)

        # Prediction
    y_predict = dtr.predict(X_test[i])

    result = np.zeros((n_trip_test, 2))

    # for i in range(n_trip_test):
    #     result[i, 0] = LATITUDE
    #     result[i, 1] = LONGITUDE
    #
    # # Write submission
    # print_submission(trip_id=trip_id, result=result, name="sampleSubmission_generated")
    
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Antonio Sutera

# How to use this function?
# trip_id is the first column of the submission, i.e., the id of the sample
# result is a two-column matrix made of rows of "Latitude"x"Longitude" pairs.
# 		Note that the line "result[i,:]" should correspond to id "trip_id[i]"
# name is the name you want for your submission file
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

if __name__ == "__main__":

    # Data Loading
    data = pd.read_csv('train_data.csv', index_col="TRIP_ID")
    n_trip_train, _ = data.shape
    print('Shape of train data: {}'.format(data.shape))

    # Test Set Loading
    test = pd.read_csv('test.csv', index_col="TRIP_ID")
    n_trip_test, _ = test.shape
    print('Shape of test data: {}'.format(test.shape))
    trip_id = list(test.index)

    # # How to make timestamp nicer
    # clean_timestamp = pd.to_datetime(data["TIMESTAMP"], unit="s")

    # Training
    # TODO
    LATITUDE = 41.146504
    LONGITUDE = -8.611317

    # Prediction
    result = np.zeros((n_trip_test, 2))

    for i in range(n_trip_test):
        result[i, 0] = LATITUDE
        result[i, 1] = LONGITUDE

    # Write submission
    print_submission(trip_id=trip_id, result=result, name="sampleSubmission_generated")

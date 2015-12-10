# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import pandas as pd
import datetime
import numpy as np

def from_time_to_day_period(row):
    st = datetime.datetime.fromtimestamp(row["TIMESTAMP"]).strftime('%H')
    hour = int(st)

    return hour  

if __name__ == "__main__":

    # Data Loading
    test = pd.read_csv('test.csv', index_col="TRIP_ID")
    n_trip_train, _ = test.shape
    print('Shape of test data: {}'.format(test.shape))

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

    test = test[test["POLYLINE"].map(len) > 2]

    # Extract 'y' and long and lat
    rides = test['POLYLINE'].values
    rides = list(map(eval, rides))

    test['POLYLINE'] = rides

    test["DIRECTION"] = 0
    # Delete all the datas which have missing data in their paths
    test = test[test["MISSING_DATA"] != 1]
    for i in range(len(rides)):
        dist = len(rides[i])

        if dist <= 1:
            continue

        #Compute the direction

        if dist > 1:
            dist1 = np.linalg.norm(np.asarray(rides[i][-2])-np.asarray(rides[i][0]))
            if dist1 == 0:
                dist1 = 0.0000001
            tmp = rides[i][-2]
            tmp[0] = rides[i][0][0]
            dist2 = np.linalg.norm(np.asarray(tmp)-np.asarray(rides[i][0]))

            tmp_angle = np.arccos(dist2/dist1)*(180/np.pi)

            if rides[i][-2][1] >= rides[i][0][1]:
                if rides[i][-2][0] >= rides[i][0][0]:
                    tmp_angle =tmp_angle
                else:
                    tmp_angle = 180-tmp_angle
            else:
                if rides[i][-2][0] >= rides[i][0][0]:
                    tmp_angle = 360-tmp_angle
                else:
                    tmp_angle = 270-tmp_angle

            test["DIRECTION"].iloc[i] = tmp_angle

    test.to_pickle('dir_test_pickle.pkl')

    print("End of conversion")

    # Data Loading
    data = pd.read_pickle('dir_test_pickle.pkl')
    n_trip_train, _ = data.shape
    print('Shape of test data: {}'.format(data.shape))


    # Data Loading
    data = pd.read_csv('train_data.csv', index_col="TRIP_ID")
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
    data = data[data["MISSING_DATA"] != 1]
    data = data[data["POLYLINE"].map(len) > 2]

    # Extract 'y' and long and lat
    rides = data['POLYLINE'].values
    rides = list(map(eval, rides))

    data['POLYLINE'] = rides

    data["DIRECTION"]=0
    # Delete all the datas which have missing data in their paths
    for i in range(len(rides)):

        dist = len(rides[i])

        if dist <= 1:
            continue

        # Compute the direction
        dist1 = np.linalg.norm(np.asarray(rides[i][-2])-np.asarray(rides[i][0]))
        if dist1 == 0:
            dist1 = 0.0000001
        tmp = rides[i][-2]
        tmp[0] = rides[i][0][0]
        dist2 = np.linalg.norm(np.asarray(tmp)-np.asarray(rides[i][0]))

        tmp_angle = np.arccos(dist2/dist1)*(180/np.pi)

        if rides[i][-2][1] >= rides[i][0][1]:
            if rides[i][-2][0] >= rides[i][0][0]:
                tmp_angle =tmp_angle
            else:
                tmp_angle = 180-tmp_angle
        else:
            if rides[i][-2][0] >= rides[i][0][0]:
                tmp_angle = 360-tmp_angle
            else:
                tmp_angle = 270-tmp_angle

        data["DIRECTION"].iloc[i] = tmp_angle

    data.to_pickle('dir_data_pickle.pkl')

    print("End of conversion")

    # # Data Loading
    # data = pd.read_pickle('data_pickle.pkl')
    # n_trip_train, _ = data.shape
    # print('Shape of train data: {}'.format(data.shape))
    # print("End of reading")
import datetime
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
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

if __name__ == "__main__":

    data_read = pd.read_csv('train_data.csv', index_col="TRIP_ID")
    data = data_read.sample(frac=0.05)
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
    data = data[data["POLYLINE"] != '[]']
    data = data[["CALL_TYPE", "TAXI_ID", "DAY_TYPE", "TIMESTAMP", "POLYLINE"]]
    # data = data[["POLYLINE"]]

    kf = KFold(len(data), n_folds=3)

    max_depths = np.arange(11, 40, 5)

    for max_depth in max_depths:
        b = []
        v = []
        e = []
        for train_index, test_index in kf:
            X_ls = data.iloc[train_index]
            rides_ls = extract_rides(X_ls)
            origins_ls, lengths_ls = extract_origins_lengths(rides_ls)
            X_ls.drop("POLYLINE",1,inplace=True)
            X_ls["org_x"] = origins_ls[:,0]
            X_ls["org_y"] = origins_ls[:,1]

            regr = DecisionTreeClassifier(max_depth=max_depth)
            regr.fit(X_ls, lengths_ls)

            X_test = data.iloc[test_index]
            rides_test = extract_rides(X_test)
            origins_test, lengths_test = extract_origins_lengths(rides_test)
            X_test.drop("POLYLINE",1,inplace=True)
            X_test["org_x"] = origins_test[:,0]
            X_test["org_y"] = origins_test[:,1]

            predict = regr.predict(X_test)

            b.append(bias(predict, lengths_test))
            v.append(var_ls(predict))
            e.append(mae(np.array(predict), np.array(lengths_test)))
        print("max_depth = {}".format(max_depth))
        print("Bias: ", np.mean(b))
        print("Variance: ", np.mean(v))
        print("Mean absolute error: ", np.mean(e))



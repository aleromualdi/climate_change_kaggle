
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


import sys
sys.path.append('../')
from models import svr, deep_nn

from numpy.random import seed
seed(1)


#Heihe, Kyzyl, Hailar, Nehe 

SPLIT_DATE = datetime.datetime(1950, 12, 1)
END_DATE = datetime.datetime(2013, 12, 1)



def get_data_city(data, city):
    
    data_city = data[data['City']==city]
    mask = data_city.index <= END_DATE
    data_city = data_city.loc[mask]

    return data_city[['AverageTemperature']]


def make_data(data):
    """ Make train and test data. Used 1-shift look back """

    # train test split
    mask = data.index <= SPLIT_DATE
    train = data.loc[mask]
    test = data.loc[~mask]

    train = train.rename(columns={'AverageTemperature': 'Y'})
    test = test.rename(columns={'AverageTemperature': 'Y'})

    train['X_1'] = train['Y'].shift(1)
    test['X_1'] = test['Y'].shift(1)

    X_train = train.dropna().drop('Y', axis=1)
    y_train = train.dropna().drop('X_1', axis=1)

    X_test = test.dropna().drop('Y', axis=1)
    y_test = test.dropna().drop('X_1', axis=1)

    X_train = X_train.values
    y_train = y_train.values

    X_test = X_test.values
    y_test = y_test.values

    print('Train size: (%d x %d)'%(X_train.shape[0], X_train.shape[1]))
    print('Test size: (%d x %d)'%(X_test.shape[0], X_test.shape[1]))

    return X_train, X_test, y_train, y_test, test.index[1:]


def predict_svr(input_data, city):

    X_train, X_test, y_train, y_test, test_dates  = input_data

    svr_model = svr()

    svr_model.fit(X_train, y_train)
    y_pred = svr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('mse: %f'%mse)

    plt.plot(test_dates , y_test, label='true')
    plt.plot(test_dates , y_pred, label='prediction')
    plt.title('SRV. mse = %.2f'%mse)
    plt.ylabel('T (Celsius)')
    plt.xlabel('Date')
    plt.legend()
    plt.savefig('../output/prediction_svr_%s.png'%city)
    plt.close()


def predict_nn(input_data, city):

    X_train, X_test, y_train, y_test, test_dates = input_data

    deep_model = deep_nn((X_test.shape[1],))
    deep_model.fit(X_train, y_train, batch_size=16, epochs=100, verbose=1)

    y_pred = deep_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('mse: %f'%mse)
    
    plt.plot(test_dates, y_test, label='true')
    plt.plot(test_dates, y_pred, label='predictions')
    plt.legend()
    plt.title('Neural newtork. mse = %.2f'%mse)
    plt.ylabel('T (Celsius)')
    plt.xlabel('Date')
    plt.savefig('../output/prediction_neural_network_%s.png'%city)
    plt.close()


def main():
    
    # load global land temperatures by city data
    df = pd.read_pickle('global_temp_city_year.pkl')

    for city in df.City.unique():
        
        print('Processing city = ', city)

        raw_data = get_data_city(df, city)

        scaler = MinMaxScaler()
        raw_data['AverageTemperature'] = scaler.fit_transform(raw_data)

        input_data = make_data(raw_data)

        print('training SVM model...')
        predict_svr(input_data, city)
        print('training neural network model...')
        predict_nn(input_data, city)


if __name__ == "__main__":
    main()
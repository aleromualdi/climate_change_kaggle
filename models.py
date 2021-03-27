from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.svm import SVR
from keras.optimizers import Adam
import keras.backend as K

def svr():
    return SVR(kernel='rbf')


def deep_nn(input_shape):

    K.clear_session()
    model = Sequential()
    model.add(Dense(50, input_shape=input_shape, activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(50, input_shape=input_shape, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
    return model
    

def lstm_nn(input_shape):
    model = Sequential()
    model.add(LSTM(4, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model
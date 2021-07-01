# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from get_data_from_mat import mat_data


# load a single file as a numpy array
def load_file():
    dataframe = np.array()
    for gest in range(0, len(mat_data)):
        dataframe.append(mat_data[gest][0])
    return dataframe


print(load_file)

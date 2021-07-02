from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# gets data from matrix

def get_train_test_sets(matrix_name):
    fil_mat_arr = loadmat(matrix_name+'.mat')
    X=fil_mat_arr[matrix_name]
    Y = np.array([[i]*30 for i in range(1,6)]).reshape(1,30*5)[0]
    # for i in range(60):
    #     X=np.delete(X,90,0)
    # Y=[]
    # for i in [1,2,5]:
    #     for j in range(30):
    #         Y.append(i)
        # Y.join([i]*30)
    # Y=np.array([[1]*30,[2]*30,[5]*30])
    Y=to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=42, stratify=Y)
    return X_train, y_train, X_test, y_test

if __name__ =="__main__":
    print(get_train_test_sets('fil_s_3mat'))



# Note: train_test_split does split evenly if stratify = Y
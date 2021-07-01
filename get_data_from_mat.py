from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# gets data from matrix

fil_mat_arr = loadmat('fil_mat.mat')
fil_mat_lst=[tuple([elem for elem in row]) for row in fil_mat_arr['fil_mat'][0:-1]]
columns = []
for gest in ["+y","+z", "read", "-y", "-z"]:
    for suff in ["_k_", "_s_"]:
        for num in range(1, 31):
            columns+=[gest+suff+str(num)]
df = pd.DataFrame(fil_mat_lst, columns=columns)

X = df.transpose().to_numpy()
Y = np.array([[i]*60 for i in range(1,6)]).reshape(1,60*5)[0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=42, stratify=Y)

# Note: train_test_split does split evenly if stratify = Y
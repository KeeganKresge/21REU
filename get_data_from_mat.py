from scipy.io import loadmat
import pandas as pd


# gets data from matrix

def mat_data():
    fil_mat_arr = loadmat('fil_mat.mat')
    fil_mat_lst = [tuple([elem for elem in row]) for row in fil_mat_arr['fil_mat'][0:-1]]
    columns = []
    for gest in ["+y", "+z", "read", "-y", "-z"]:
        for suff in ["_k_", "_s_"]:
            for num in range(1, 31):
                columns += [gest + suff + str(num)]
    df = pd.DataFrame(fil_mat_lst, columns=columns)
    return df


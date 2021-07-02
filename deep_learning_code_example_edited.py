# lstm model
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from get_data_from_mat import get_train_test_sets
from matplotlib import pyplot
import winsound



# # load a single file as a numpy array
# def load_file(filepath):
#     dataframe = read_csv(filepath, header=None, delim_whitespace=True)
#     return dataframe.values
#
#
# # load a list of files and return as a 3d numpy array
# def load_group(filenames, prefix=''):
#     loaded = list()
#     for name in filenames:
#         # use matlab data
#         data = mat_data
#         loaded.append(data)
#     # stack group so that features are the 3rd dimension
#     loaded = dstack(loaded)
#     return loaded
#
#
# # load a dataset group, such as train or test
# def load_dataset_group(group, prefix=''):
#     filepath = prefix + group + '/Inertial Signals/'
#     # load all 9 files as a single array
#     filenames = list()
#     # total acceleration
#     filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
#     # body acceleration
#     filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
#     # body gyroscope
#     filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
#     # load input data
#     X = load_group(filenames, filepath)
#     # load class output
#     y = load_file(prefix + group + '/y_' + group + '.txt')
#
#     # X is matrix y is names
#
#     return X, y
#
#
#
#
# # load the dataset, returns train and test X and y elements
# def load_dataset(prefix=''):
#     # load all train
#     trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
#     print(trainX.shape, trainy.shape)
#     # load all test
#     testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
#     print(testX.shape, testy.shape)
#     # zero-offset class values
#     trainy = trainy - 1
#     testy = testy - 1
#     # one hot encode y
#     trainy = to_categorical(trainy)
#     testy = to_categorical(testy)
#     print(trainX.shape, trainy.shape, testX.shape, testy.shape)
#     return trainX, trainy, testX, testy


# fit and evaluate a model
def make_model(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# fit and evaluate a model
def make_model2(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv2D(3, (3, 3), padding='valid', activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(3, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(3, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def evaluate_model(model, testX, testy):
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

def get_confusion_matrix(model, testX, testy):
    y_pred = model.predict_classes(testX)
    testy=[np.argmax(elem, axis=-1) for elem in testy]
    cm = confusion_matrix(testy, y_pred)
    # print('Classification Report')
    # target_names = ['Cats', 'Dogs', 'Horse']
    # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
    return cm


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_accuracy_test(repeats=50):
    # repeat experiment
    scores = list()
    for r in range(repeats):
        model = make_model(trainX, trainy)
        score = evaluate_model(model, trainX, trainy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

def run_confusion_matrix_test(repeats=50):
    # repeat confusion matrix
    cm = [[0 * 5] * 5]
    for r in range(repeats):
        print(r)
        model = make_model(trainX, trainy)
        cm+= get_confusion_matrix(model, testX, testy)
    print("Confusion Matrix:\n",cm)


verbose, epochs, batch_size = 0, 15, 64
# load data
trainX, trainy, _, _ = get_train_test_sets('unf_s_3mat')
_, _, testX, testy   = get_train_test_sets('unf_s_3mat')

# run the experiment
# run_accuracy_test()

# get confusion matrix
run_confusion_matrix_test()

winsound.Beep(300, 2000)

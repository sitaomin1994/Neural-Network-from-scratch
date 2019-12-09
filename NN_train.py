import numpy as np
import csv
import sys
import os
from MLP import MLPClassifier

from sklearn.metrics import accuracy_score

import mnist

def load_data(train_file, test_file):
    training_data = np.array(list(csv.reader(open(train_file, "r"), delimiter=",")))
    testing_data = np.array(list(csv.reader(open(test_file, "r"), delimiter=",")))

    X_train = training_data[:, :-1].astype("float")
    y_train = training_data[:, -1].astype("int")
    X_test = testing_data[:].astype("float")

    return X_train, y_train, X_test


if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    print("Usage: python NeuralNetwork.py training_data_path testing_data_path")
    #    sys.exit()

    # load data
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]
    # train_file = 'data/blackbox13_train.csv'
    # test_file = 'data/blackbox13_test.csv'

    # X_train, y_train, X_test = load_data(train_file, test_file)
    # y_test = np.array(list(csv.reader(open('data/blackbox13_example_predictions.csv', "r"), delimiter=",")))[:].astype(
    #     "int")

    mnist.init()

    X_train, y_train, X_test, y_test = mnist.load()


    clf = MLPClassifier(hidden_layer_sizes= (64,32, ), epoches= 100, verbose = True, batch_size = 256,
                        random_state = 1, solver = 'adam', learning_rate = 0.001, early_stopping= False,
                        momentum = 0.9, nesterovs_momentum = False)

    clf.fit(X_train, y_train, evaluation_set={'X_eval':X_test,'y_eval':y_test})
    clf.plot_graph()

    y_pred = clf.predict(X_test)

    print(accuracy_score(y_pred, y_test))

    pred_file = os.path.basename(train_file).split("_")[0] + "_predictions.csv"
    y_test_pred = clf.predict(X_test)
    with open(pred_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerows([y] for y in y_test_pred)
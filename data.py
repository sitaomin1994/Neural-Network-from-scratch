import numpy as np
import csv

from Blackbox11 import blackbox11
from Blackbox12 import blackbox12
from Blackbox13 import blackbox13



def generate_data(x_range, y_range, z_range, blackbox, file_name):
    """ generate data, and randomly split data into 3 parts (5 files): 
        training data 60%, testing data 20%, hidden testing data 20%
        both testing data and hidden testing data are split into X and y
    """
    lines = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                if blackbox == 1:
                    w = blackbox11.ask(x, y, z)
                elif blackbox == 2:
                    w = blackbox12.ask(x, y, z)
                elif blackbox == 3:
                    w = blackbox13.ask(x, y, z)
                line = [x, y, z, w]
                lines.append(line)

    data = np.array(list(lines))

    np.random.shuffle(data)
    len_ = len(data)
    with open(file_name + "_train.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data[: int(len_ * 0.6)]) 
    with open(file_name + "_test.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data[ int(len_ * 0.6): int(len_ * 0.8), :-1])
    with open(file_name + "_example_predictions.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows([y] for y in data[ int(len_ * 0.6): int(len_ * 0.8):, -1])
    with open(file_name + "_hidden_test.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data[int(len_ * 0.8):, :-1])
    with open(file_name + "_hidden_label.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows([y] for y in data[int(len_ * 0.8):, -1])



if __name__ == "__main__":
    generate_data(np.arange(1, 51), np.arange(1, 51), np.arange(1,51), 1,  "./data/blackbox11")
    generate_data(np.arange(1, 51), np.arange(1, 51), np.arange(1,51), 2,  "./data/blackbox12")
    generate_data(np.arange(1, 51), np.arange(1, 51), np.arange(1,51), 3,  "./data/blackbox13")


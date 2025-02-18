import numpy as np
import pandas as pd

FILE_NAME = "dataset_train.csv"


def sigmoid(z):
    """Ca marche c cool"""
    return (1/(1 + np.exp(-z)))


def logistic_regression(X, y, epoch, learning_rate):
    m, n = X.shape  # m = lines and n = column
    unique_houses = np.unique(y)
    nb_class = len(unique_houses) # get all uniques classes from y
    weight_tab = np.zeros((nb_class, n))
    # replace houses by 1 and 0 in the code for each unique houses (y_i being a double array)
    for i, house in enumerate(unique_houses):
        y_i = np.where(np.array(y) == house, 1, 0) # convert the array y in np.array to apply np func on it
        for _ in range(epoch):
            z = np.dot(X, weight_tab[i])
            y_p = sigmoid(z)
            weight_diff = 1/m * (np.dot(X.T, (y_p - y_i)))
            weight_tab[i] -= learning_rate * weight_diff
    return weight_tab


if __name__ == "__main__":
    try : 
        data = pd.read_csv(FILE_NAME)
        epoch = 1000
        house_pairs = [["History of Magic", "History of Magic"], # Gryffindor
                    ["Charms", "Charms"], # Ravenclaw
                    ["Divination", "Divination"], # Slytherin
                    ["Astronomy", "Herbology"]] # Hufflepuff
        learning_rate = 0.1
        # X = data.select_dtypes(include=[np.number]).values
        X = data[["Charms", "Charms"]].values
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) # Normalize data
        print(logistic_regression(X, data["Hogwarts House"].values, epoch, learning_rate))
    except:
        print("An exception occurred")
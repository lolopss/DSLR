import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler

FILE_NAME = "dataset_train.csv"

def sigmoid(z):
    """Apply sigmoid function while preventing overflow"""
    # z = np.clip(z, -500, 500)  # Prevents numerical instability
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, epochs, learning_rate):
    m, n = X.shape  # m = number of samples, n = number of features
    unique_houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
    nb_class = len(unique_houses)  # Number of unique classes (Hogwarts houses)

    weight_tab = np.zeros((nb_class, n))  # Initialize weights for each class
    bias = np.zeros(nb_class)  # Initialize bias for each class

    # One-vs-all training
    for i, house in enumerate(unique_houses):
        y_i = np.where(np.array(y) == house, 1, 0)  # Convert y into binary (1 for the current class, 0 otherwise)

        for _ in range(epochs):
            z = np.dot(X, weight_tab[i]) + bias[i]  # Linear combination with bias
            y_p = sigmoid(z)  # Apply sigmoid activation

            # Compute gradients
            weight_diff = (1/m) * np.dot(X.T, (y_p - y_i))  # Gradient for weights
            bias_diff = (1/m) * np.sum(y_p - y_i)  # Gradient for bias

            # Update weights and bias
            weight_tab[i] -= learning_rate * weight_diff
            bias[i] -= learning_rate * bias_diff

    return weight_tab, bias


def predict(X, weights, bias):
    """Compute probabilities and return the predicted house"""
    predictions = []
    for student in X:
        z = np.dot(student, weights.T) + bias  # Linear combination with bias
        probabilities = sigmoid(z)
        predictions.append(np.argmax(probabilities, axis=0))  # Get class with highest probability
    return predictions


def compute_accuracy(true_labels, predicted_labels):
    """Calculate accuracy percentage"""
    correct = sum(1 for i in range(len(true_labels)) if true_labels[i] == predicted_labels[i])
    return (correct / len(true_labels)) * 100


if __name__ == "__main__":
    try:
        data = pd.read_csv(FILE_NAME)

        # Check for missing values before dropping them
        print("Missing values before dropna():", data.isnull().sum().sum())

        epochs = 210
        learning_rate = 10

        # Select features to use
        pair = ["Astronomy", "Herbology"]
        data = data.dropna(subset = pair)  # Remove rows with missing values
        X = data[pair].values

        # Normalize data using MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)  # Store as NumPy array

        test = {"Gryffindor":0, "Ravenclaw":1, "Slytherin":2, "Hufflepuff":3}
        y = data["Hogwarts House"].values
        t = [test[house] for house in y]
        # Train the model
        weight_tab, bias = logistic_regression(X, y, epochs, learning_rate)

        print("Weights:\n", weight_tab)
        print("Bias:\n", bias)
        predictions = predict(X, weight_tab, bias)
        print(compute_accuracy(t, predictions))


        # Save parameters to a CSV file
        with open('thetas.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((weight_tab, bias.reshape(-1, 1))))  # Reshape bias to avoid shape mismatch

    except Exception as e:
        print("An exception occurred:", e)

from logreg_train import sigmoid
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler

DATASET = "dataset_train.csv"

# Define the order of the classes (houses)
house_names = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]

def predict(X, weights, bias):
    """Compute probabilities and return the predicted house"""
    predictions = []
    for student in X:
        z = np.dot(student, weights.T) + bias  # Linear combination with bias
        probabilities = sigmoid(z)
        predictions.append(np.argmax(probabilities, axis=0))  # Get class with highest probability
    return predictions

if __name__ == "__main__":
    try:
        # Load test dataset
        test_data = pd.read_csv(DATASET)
        test_data = test_data.dropna()
        # Select the same features used during training
        X = test_data[["Astronomy", "Herbology"]].values
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X) # Normalize data

        # Load trained weights and bias
        with open('thetas.csv', mode='r') as file:
            reader = csv.reader(file)
            theta_data = np.array([list(map(float, row)) for row in reader])

        weights = theta_data[:, :-1]  # All columns except last one
        bias = theta_data[:, -1]  # Last column
        print(weights)
        print(bias)

        # Make predictions
        predictions = predict(X, weights, bias)

        # Map predictions to house names
        house_predictions = [house_names[p] for p in predictions]

        # Save predictions to houses.csv
        with open("houses.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Hogwarts House"])  # Header
            for i, house in enumerate(house_predictions):
                writer.writerow([i, house])

        print("Predictions saved to houses.csv")

    except Exception as e:
        print("An error occurred:", e)
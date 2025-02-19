import pandas as pd
import sys

def compute_accuracy(true_labels, predicted_labels):
    """Calculate accuracy percentage"""
    correct = sum(1 for i in range(len(true_labels)) if true_labels[i] == predicted_labels[i])
    return (correct / len(true_labels)) * 100

DATASET_TEST = "dataset_train.csv"
PREDICTIONS = "houses.csv"

if __name__ == "__main__":
    try:
        # Load actual labels from dataset_test.csv
        test_data = pd.read_csv(DATASET_TEST)
        test_data = test_data.dropna()  
        if "Hogwarts House" not in test_data.columns:
            print("Error: The test dataset does not contain actual house labels.")
            sys.exit(1)

        actual_labels = test_data["Hogwarts House"].tolist()  # Convert to list

        # Load predicted labels from houses.csv
        predicted_data = pd.read_csv(PREDICTIONS)
        predicted_labels = predicted_data["Hogwarts House"].tolist()  # Convert to list

        # Ensure both files have the same number of rows
        if len(actual_labels) != len(predicted_labels):
            print(f"Error: Expected {len(actual_labels)} predictions, but got {len(predicted_labels)}.")
            sys.exit(1)

        # Compute accuracy
        accuracy = compute_accuracy(actual_labels, predicted_labels)
        print(f"Model Accuracy: {accuracy:.2f}%")

    except Exception as e:
        print("An error occurred:", e)

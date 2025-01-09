import sys
import csv
import numpy as np

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            data.append(row)
    return headers, data

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def ft_mean(column):
    # dont use the numpy mean function
    mean = [float(x) for x in column if is_number(x)]
    print(mean)
    return mean


def calculate_statistics(column):
    column = np.array(column, dtype=float)
    count = len(column)
    mean = np.mean(column)
    std = np.std(column)
    min_val = np.min(column)
    q25 = np.percentile(column, 25)
    median = np.median(column)
    q75 = np.percentile(column, 75)
    max_val = np.max(column)
    return count, mean, std, min_val, q25, median, q75, max_val

def describe_dataset(file_path):
    headers, data = read_csv(file_path)
    numerical_data = {header: [] for header in headers}

    for row in data:
        for header, value in zip(headers, row):
            if is_number(value):
                numerical_data[header].append(value)
    ft_mean(numerical_data['Arithmancy'])
    print(f"{'Feature':<30} | {'Count':<10} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'25%':<10} | {'50%':<10} | {'75%':<10} | {'Max':<10}")
    for header, values in numerical_data.items():
        if values:
            count, mean, std, min_val, q25, median, q75, max_val = calculate_statistics(values)
            print(f"{header:<30} | {count:<10.2f} | {mean:<10.2f} | {std:<10.2f} | {min_val:<10.2f} | {q25:<10.2f} | {median:<10.2f} | {q75:<10.2f} | {max_val:<10.2f}")

if len(sys.argv) != 2:
    print("Usage: python describe.py <dataset.csv>")
    sys.exit(1)

file_path = sys.argv[1]
describe_dataset(file_path)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_numerical_data(df):
    numerical_data = df.select_dtypes(include=[np.number])
    return numerical_data

def plot_histograms(data):
    valid_data = {course: data[course].dropna().values for course in data.columns if data[course].notna().any()}
    num_courses = len(valid_data.keys())
    num_cols = 3
    num_rows = (num_courses + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    axes = axes.flatten()

    for i, (course, values) in enumerate(list(valid_data.items())[1:]):  # Skip the first item
        axes[i].hist(values, bins=30, edgecolor='black')
        axes[i].set_title(f'Histogram of {course}', fontsize=10)
        axes[i].set_xlabel(course, fontsize=8)
        axes[i].set_ylabel('Frequency', fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.6)
    plt.show()

if __name__ == "__main__":
    dataset_train = 'dataset_train.csv'
    df = pd.read_csv(dataset_train)
    numerical_data = df.select_dtypes(include=[np.number])

    # Plot histograms for all courses in one window
    plot_histograms(numerical_data)

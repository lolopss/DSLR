import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
data = pd.read_csv('dataset_train.csv')

def extract_numerical_data(df):
    # Select only numerical columns
    numerical_data = df.select_dtypes(include=[np.number])
    return numerical_data

def find_most_similar_features(df):
    features = df.columns
    max_corr = -1
    feature_pair = (None, None)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            # Drop NaN values and ensure common data points
            common_data = df[[features[i], features[j]]].dropna()
            if len(common_data) > 0:
                # A value of -1 meaning a total negative linear correlation, 0 being no correlation, 
                # and + 1 meaning a total positive correlation.
                corr, _ = pearsonr(common_data[features[i]], common_data[features[j]])
                print(f"Correlation between {features[i]} and {features[j]}: {corr}")
                if abs(corr) > max_corr:
                    max_corr = abs(corr)
                    feature_pair = (features[i], features[j])
    print(f"Max Correlation : {max_corr}")
    return feature_pair

def plot_scatter(df, feature1, feature2, houses):
    # Filter out NaN values and ensure common data points
    common_data = df[[feature1, feature2, houses]].dropna()
    
    if not common_data.empty:
        x = common_data[feature1]
        y = common_data[feature2]
        house_colors = common_data[houses].map({
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow',
            'Ravenclaw': 'blue',
            'Slytherin': 'green'
        })
        
        # Create the scatter plot
        plt.scatter(x, y, c=house_colors, alpha=0.5)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    else:
        print(f"No valid data to plot for {feature1} and {feature2}")

if __name__ == "__main__":
    try:
        # Extract numerical data
        numerical_data_train = extract_numerical_data(data)

        # Find the most similar features
        feature1, feature2 = find_most_similar_features(numerical_data_train)

        if feature1 and feature2:
            print(f"The most similar features are: {feature1} and {feature2}")
            plot_scatter(data, feature1, feature2, 'Hogwarts House')
        else:
            print("No similar features found.")
    except Exception as e:
        print(f"An error occurred: {e}")
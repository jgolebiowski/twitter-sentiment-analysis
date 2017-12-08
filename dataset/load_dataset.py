"""Loading and operating on the dataset
data from: https://data.world/crowdflower/sentiment-analysis-in-text
"""
import pickle
from text_preprocessor import *
from csv_helper import *

if (__name__ == "__main__"):
    final_data, final_labels, labels2names = load_csv2lists("text_emotion.csv")

    for index in range(len(final_data)):
        final_data[index] = cleanup_string(final_data[index])

    for i in range(20):
        print(labels2names[final_labels[i]], final_data[i])
    pickle_object((final_data, final_labels, labels2names), "dataset_pickled.pkl")

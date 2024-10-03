import numpy as np
from sklearn.model_selection import KFold
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from random import sample
from scipy.sparse import csr_matrix, issparse
import re
from sklearn.metrics.pairwise import pairwise_distances


class myKNNClassifier:
    def __init__(self, nn, dist_metric):
        self.num_of_neighbors = nn
        self.metric = dist_metric

    def trainClassifier(self, train_data_dist_matrix, train_data_y_matrix):
        self.train_data_dist_matrix = train_data_dist_matrix
        self.train_data_y_matrix = train_data_y_matrix

    def classify_test_matrix(self, test_train_data_distance_matrix):
        predicted_classifications = []
        for test_data in range(test_train_data_distance_matrix.shape[0]):
            distance_from_all_train_data = test_train_data_distance_matrix[test_data, :]

            k_nearest_training_indexes = np.argsort(distance_from_all_train_data)[:self.num_of_neighbors]
            k_nearest_training_labels = self.train_data_y_matrix[k_nearest_training_indexes]

            most_frequent_label = Counter(k_nearest_training_labels).most_common(1)[0][0]
            predicted_classifications.append(most_frequent_label)

        return np.array(predicted_classifications)


def perform_cross_validation(train_data, y_data, nn, full_distance_matrix, number_of_splits, dist_metric):
    k_fold = KFold(n_splits=number_of_splits, shuffle=True, random_state=42)
    validation_accuracy_list = []

    for training_index, validation_index in k_fold.split(train_data):
        y_training = y_data[training_index]
        y_validation = y_data[validation_index]

        train_train_distance_matrix = full_distance_matrix[np.ix_(training_index, training_index)]
        train_validation_distance_matrix = full_distance_matrix[np.ix_(validation_index, training_index)]

        classifier = myKNNClassifier(nn, dist_metric)
        classifier.trainClassifier(train_train_distance_matrix, y_training)

        predictions = classifier.classify_test_matrix(train_validation_distance_matrix)

        # Calculate accuracy
        validation_accuracy = np.mean(predictions == y_validation)
        validation_accuracy_list.append(validation_accuracy)

    return np.mean(validation_accuracy_list)


def exhaustive_parameter_search(train_tfIDF_matrix, y_data, k_list, dist_matric_list, number_of_splits, k_chi_list):
    print("Doing validation....")
    # Define the range of k-values and distance metrics to sample from
    best_parameters = None
    best_accuracy = 0
    parameter_stats_list = []

    for k_chi in k_chi_list:
        print(f"Applying SelectKBest with k={k_chi}...")
        chi2_selector = SelectKBest(chi2, k=k_chi)
        selected_train_tfIDF_matrix = chi2_selector.fit_transform(train_tfIDF_matrix, train_rating_df)
        dist_metric_dict = {}
        for each_metric in dist_matric_list:
            dist_metric_dict[each_metric] = pairwise_distances(selected_train_tfIDF_matrix, selected_train_tfIDF_matrix,
                                                               metric=each_metric)
        print(
            f"SelectKBest applied. Shape of reduced matrix: {selected_train_tfIDF_matrix.shape}, Sparse Format: {issparse(selected_train_tfIDF_matrix)}")
        # full_train_train_distance_matrix_euclidean = pairwise_distances(selected_train_tfIDF_matrix, selected_train_tfIDF_matrix, metric="euclidean")
        # full_train_train_distance_matrix_cosine = pairwise_distances(selected_train_tfIDF_matrix, selected_train_tfIDF_matrix, metric="cosine")
        for k in k_list:
            for metric in dist_matric_list:
                print("evaluating for k =", k, ", metric =", metric)

                current_accuracy = perform_cross_validation(selected_train_tfIDF_matrix, y_data, k,
                                                            dist_metric_dict[metric],
                                                            number_of_splits, metric)

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_parameters = {'k_chi': k_chi, 'k': k, 'metric': metric}

                parameter_stats_list.append({'k_chi': k_chi, 'k': k, 'metric': metric, 'accuracy': current_accuracy})

    # Print the best hyperparameters and the corresponding accuracy
    print(f"Best Parameters: {best_parameters}")
    print(f"Best Cross-validated Accuracy: {best_accuracy * 100:.2f}%")

    # Optionally, you can print all results to see the performance of each combination
    print("\nAll Results:")
    # Assuming `k_chi` and `all_results` are already defined in your code
    with open("performance_log.txt", "a") as file:  # Open the file in append mode
        for result in parameter_stats_list:
            # Format the string as you want to print
            line_to_append = f"k_chi={result['k_chi']} k={result['k']}, metric={result['metric']}, accuracy={result['accuracy'] * 100:.2f}%"
            print(line_to_append)  # Print to console (optional)
            file.write(line_to_append + "\n")  # Append the line to the file with a newline character

    return parameter_stats_list, best_parameters, best_accuracy


def predict_test_data(train_tfIDF_matrix, test_review_df, train_rating_df, optimal_k, optimal_metric, optimal_k_chi,
                      prediction_file):
    print("predicting test reviews...........")
    y_data = train_rating_df.values
    chi2_selector = SelectKBest(chi2, k=optimal_k_chi)
    selected_train_tfIDF_matrix = chi2_selector.fit_transform(train_tfIDF_matrix, train_rating_df)
    test_tfIDF_matrix = tf_idf_vectorizer.transform(test_review_df)
    selected_test_tfIDF_matrix = chi2_selector.transform(test_tfIDF_matrix)
    print(
        f"SelectKBest applied. Train Shape: {selected_train_tfIDF_matrix.shape}, Test Shape: {selected_test_tfIDF_matrix.shape}")

    # Confirm the matrices are in sparse format
    print(
        f"Sparse Format Check - Train: {issparse(selected_train_tfIDF_matrix)}, Test: {issparse(selected_test_tfIDF_matrix)}")
    train_test_distance_matrix = pairwise_distances(selected_test_tfIDF_matrix, selected_train_tfIDF_matrix,
                                                    metric=optimal_metric)
    classifier = myKNNClassifier(dist_metric=optimal_metric, nn=optimal_k)
    train_train_distance_matrix = pairwise_distances(selected_train_tfIDF_matrix, selected_train_tfIDF_matrix,
                                                     metric=optimal_metric)
    classifier.trainClassifier(train_train_distance_matrix, y_data)
    predictions = classifier.classify_test_matrix(train_test_distance_matrix)

    with open(prediction_file, 'w') as f:
        for each_prediction in predictions:
            if each_prediction == 1:
                f.write("+1\n")
            else:
                f.write("-1\n")
    print(f"Predictions written to {prediction_file}")


# next steps to get better results -
# what to do with original hyphenated wors which got smooshed into one word
# change parameters of TFIDF tokenizer
# think about spelling corrections
# change bag of words approach
# change the implementation of pairwise_distances
# change the fold-number in kFold and find out if that changes anything


if __name__ == '__main__':
    full_cleaned_train_df = pd.read_csv("cleaned_train_csv.csv")
    cleaned_train_df = full_cleaned_train_df  # Using the full dataset
    train_review_df = cleaned_train_df['review']
    train_rating_df = cleaned_train_df['rating']

    full_cleaned_test_df = pd.read_csv("cleaned_test_csv.csv")
    test_review_df = full_cleaned_test_df['review']

    # finding out the number of unique words in the corpus
    # (change 'text' to your actual column name if different)
    all_text = " ".join(cleaned_train_df['review'].astype(str))  # Concatenate all rows into a single string

    # Tokenize and preprocess the text
    # Remove punctuation, convert to lowercase, and split into words
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Find unique words
    unique_words = set(words)
    print("number of unique words: ", len(unique_words))

    # Convert reviews to sparse TF-IDF matrix without converting it to dense format
    tf_idf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=len(unique_words),  # Use a large number of features
        min_df=2,
        max_df=0.8,
        stop_words='english',
        norm='l2'
    )

    print("Creating sparse TF-IDF matrix...")
    train_tfIDF_matrix = tf_idf_vectorizer.fit_transform(train_review_df)  # Do not use .toarray()
    print(f"TF-IDF matrix created. Shape: {train_tfIDF_matrix.shape}, Sparse Format: {issparse(train_tfIDF_matrix)}")

    k_list = [140, 150]
    chi_k_list = [25000]
    metric_list = ['cosine', 'euclidean']

    exhaustive_parameter_search(train_tfIDF_matrix, train_rating_df.values, k_list, metric_list, 5, chi_k_list)

    """
    print("Doing test.............")
    optimal_k_chi = 35000
    optimal_k = 110
    optimal_metric = 'cosine'
    prediction_file = 'test_predictions.txt'
    predict_test_data(train_tfIDF_matrix, test_review_df, train_rating_df,optimal_k_chi = optimal_k_chi,
                      optimal_k=optimal_k, optimal_metric=optimal_metric, prediction_file=prediction_file)

    """











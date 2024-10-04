import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
import string
from tqdm import tqdm
from joblib import Parallel, delayed
import nltk
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from num2words import num2words

import numpy as np
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from random import sample
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import pairwise_distances


def convert_text_to_csv(input_file,output_csv, is_train_file = True):
    if is_train_file:
        # Open input file for reading and output file for writing, with utf-8 encoding
        with open(input_file, 'r', encoding='utf-8') as in_file, open(output_csv, 'w', newline='',
                                                                      encoding='utf-8') as out_csv:
            csv_writer = csv.writer(out_csv)

            # Write the header for the CSV file
            csv_writer.writerow(['review', 'rating'])

            # Process each line in the input file
            for line in in_file:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line.endswith("#EOF"):
                    rating, review = line.split("\t", 1)  # Split on the first tab to separate label and text
                    review = review[:-4].strip()  # Remove the '#EOF' from the text
                    csv_writer.writerow([review, rating])  # Write to CSV
    else:
        # Read the text file
        with open(input_file, 'r', encoding='utf-8') as in_file:
            test_review_corpus = in_file.read()

        # Split the content by #EOF
        test_reviews = test_review_corpus.split('#EOF')

        # Remove empty strings and strip unnecessary whitespace
        test_reviews = [test_review.strip() for test_review in test_reviews if test_review.strip()]

        # Create a DataFrame with a single column 'test_review'
        df = pd.DataFrame(test_reviews, columns=['test_review'])

        # Save the DataFrame to a CSV file
        df.to_csv(output_csv, index=False)
    print("CSV file created successfully.")



# Function to remove HTML tags from a given text
def remove_html_tags(review):
    soup = BeautifulSoup(review, "html.parser")
    return soup.get_text()


# Function to remove URLs
def remove_urls(review):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', review)


# Function to remove non-alphanumeric characters
def remove_non_an_chars(review):
    refined_review = re.sub(r'[^a-zA-Z0-9\s]', '', review)
    return refined_review


# Function to remove punctuation
def remove_punctuation(review):
    refined_review = review.translate(str.maketrans("", "", string.punctuation))
    return refined_review


# Function to replace numbers with words
def replace_number_with_word(review):
    review_words = []
    for word in review.split():
        if word.isdigit():
            review_words.append(num2words(int(word)))
        else:
            review_words.append(word)
    refined_review = ' '.join(review_words)
    return refined_review


# Function to remove and manage whitespace
def remove_whitespace(review):
    refined_review = re.sub(r"\s\s+", " ", review)
    return refined_review


# Function to make all text lowercase
def make_lowercase(review):
    review = review.lower()
    return review


# Function to remove stopwords
def remove_stopwords(review):
    words = word_tokenize(review)
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    return ' '.join(filtered_words)


# Spelling correction (CPU-bound)
def correct_spelling(review):
    spell = SpellChecker()
    corrected_review_words = []
    try:
        for word in review.split():
            corrected_word = spell.correction(word)
            if corrected_word is None:
                corrected_review_words.append(word)
            else:
                corrected_review_words.append(corrected_word)

        refined_review = ' '.join(corrected_review_words)
        return refined_review
    except:
        print("Some issue in spell correction")
        return review


# Preprocessing function that applies all the steps
def preprocess_review(review):
    review = remove_html_tags(review)
    review = remove_urls(review)
    review = remove_non_an_chars(review)
    review = remove_punctuation(review)
    review = replace_number_with_word(review)
    review = remove_whitespace(review)
    review = make_lowercase(review)
    review = remove_stopwords(review)
    # review = correct_spelling(review)
    return review


# Reading and processing the CSV file in batches
def process_csv_in_batches(unprocessed_csv, processed_csv, batch_size=5000, is_train_csv=True):
    for i, batch in enumerate(tqdm(pd.read_csv(unprocessed_csv, chunksize=batch_size))):

        if is_train_csv:
            reviews_batch = batch['review'].tolist()
        else:
            reviews_batch = batch['test_review'].tolist()

        # Apply parallel processing on each individual review in the batch
        processed_reviews = Parallel(n_jobs=-1)(delayed(preprocess_review)(review) for review in reviews_batch)

        # Update the chunk with processed reviews
        if (is_train_csv):
            batch['review'] = processed_reviews

            # Rename the column to "review" instead of "test_review"
            batch.rename(columns={'review': 'review'}, inplace=True)

        else:
            batch['test_review'] = processed_reviews

            # Rename the column to "review" instead of "test_review"
            batch.rename(columns={'test_review': 'review'}, inplace=True)

        # For the first chunk, write with headers; for others, append without headers
        if i == 0:
            batch.to_csv(processed_csv, index=False, mode='w')  # Write the first chunk (with headers)
        else:
            batch.to_csv(processed_csv, index=False, mode='a', header=False)  # Append subsequent chunks (no headers)



def preprocess_file(original_train_csv, original_test_csv):

    processed_test_csv_file = 'cleaned_test_csv.csv'

    processed_train_csv_file = 'cleaned_train_csv.csv'

    batch_size = 5000  # Adjust the chunk size as necessary

    process_csv_in_batches(original_train_csv, processed_train_csv_file, batch_size, is_train_csv=True)
    process_csv_in_batches(original_test_csv, processed_test_csv_file, batch_size, is_train_csv=False)


class myKNNClassifier:
    def __init__(self, nn, dist_metric):
        self.num_of_neighbors = nn
        self.metric = dist_metric

    def trainClassifier(self, train_data_y_matrix):
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

        # train_train_distance_matrix = full_distance_matrix[np.ix_(training_index, training_index)]
        train_validation_distance_matrix = full_distance_matrix[np.ix_(validation_index, training_index)]

        classifier = myKNNClassifier(nn, dist_metric)
        classifier.trainClassifier(y_training)

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

                line_to_append = f"k_chi={k_chi} k={k}, metric={metric}, accuracy={current_accuracy * 100:.2f}%"
                print(line_to_append)

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

    classifier.trainClassifier(y_data)
    predictions = classifier.classify_test_matrix(train_test_distance_matrix)

    with open(prediction_file, 'w') as f:
        for each_prediction in predictions:
            if each_prediction == 1:
                f.write("+1\n")
            else:
                f.write("-1\n")
    print(f"Predictions written to {prediction_file}")


# next steps to get better results -
# gather everything (preprocess + main code) into one file
# prepare the report
# use other bag of words method
# go through the project description
# what to do with original hyphenated words which got smooshed into one word
# change parameters of TFIDF tokenizer
# think about spelling corrections
# change bag of words approach
# change the implementation of pairwise_distances
# change the fold-number in kFold and find out if that changes anything


# Main execution
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

if __name__ == '__main__':
    convert_text_to_csv("train_text.txt","train_csv.csv")
    convert_text_to_csv("test_txt.txt","test_csv.csv",False)
    print("Text to csv conversion done")
    preprocess_file("train_csv.csv","test_csv.csv")
    print("preprocessing done")

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



    k_list = [140, 150, 160,170,180]
    chi_k_list = [35000, 40000, 45000,50000,55000]
    metric_list = ['cosine']

    exhaustive_parameter_search(train_tfIDF_matrix, train_rating_df.values, k_list, metric_list, 5, chi_k_list)

    """
    print("Doing test.............")
    optimal_k_chi = 45000
    optimal_k = 150
    optimal_metric = 'cosine'
    prediction_file = 'test_predictions.txt'
    predict_test_data(train_tfIDF_matrix, test_review_df, train_rating_df,optimal_k_chi = optimal_k_chi,
                      optimal_k=optimal_k, optimal_metric=optimal_metric, prediction_file=prediction_file)
    """




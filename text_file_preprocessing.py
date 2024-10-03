import pandas as pd
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
    #review = correct_spelling(review)
    return review

# Reading and processing the CSV file in batches
def process_csv_in_batches(unprocessed_csv, processed_csv, batch_size=5000,is_train_csv = True):
    for i, batch in enumerate(tqdm(pd.read_csv(unprocessed_csv, chunksize=batch_size))):

        if is_train_csv:
            reviews_batch = batch['review'].tolist()
        else:
            reviews_batch = batch['test_review'].tolist()

        # Apply parallel processing on each individual review in the batch
        processed_reviews = Parallel(n_jobs=-1)(delayed(preprocess_review)(review) for review in reviews_batch)

        # Update the chunk with processed reviews
        if(is_train_csv):
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

# Main execution
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
if __name__ == '__main__':
    """
    unprocessed_csv_file = 'test_csv.csv'
    
    processed_csv_file = 'cleaned_test_csv.csv'
    """
    unprocessed_csv_file = 'train_csv.csv'

    processed_csv_file = 'cleaned_train_csv.csv'

    batch_size = 5000  # Adjust the chunk size as necessary

    process_csv_in_batches(unprocessed_csv_file, processed_csv_file, batch_size, is_train_csv = True)

import re
import sys

from config import Preproc


def __filter_non_english_words(text):
    pattern = re.compile(r"[^\x00-\x7F]+")
    return " ".join([word for word in str(text).split() if not bool(pattern.match(word))])


# Function to remove content inside square brackets
def __remove_square_brackets(text):
    return re.sub(r"\[.*?\]", "", text)

# Remove special characters
def __remove_special_characters(text):
    return re.sub(r"[^a-zA-Z0-9\s\?\!\']", "", text)


def __remove_url(text):
    pattern = r'https?://\S+|www\.\S+'
    return re.sub(pattern, '', text)


def __remove_number(text):
    return re.sub(r'\d+', '', text)


def __lower_case(text):
    return text.lower()


def __remove_space(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(' +', ' ', text)
    return text.strip()


def preprocess_data(text):
    text = __remove_url(text)
    text = __filter_non_english_words(text)
    text = __remove_space(text)
    text = __remove_square_brackets(text)
    text = __remove_special_characters(text)
    text = __remove_number(text)
    text = __lower_case(text)
    return text


def get_preprocessed_df(df):
    print("cleaning and reprocessing data")
    df[Preproc.text_col] = df[Preproc.text_col].apply(preprocess_data)
    df = df.dropna()
    df = df.drop_duplicates()
    df.drop_duplicates(subset=Preproc.text_col, keep='first', inplace=True)

    return df

def remap_class(df, column, remaplist):
    df[column] = df[column].map(remaplist)
    return df

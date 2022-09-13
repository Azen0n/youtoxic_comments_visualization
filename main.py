import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from visualization import plot_word_frequency, plot_wordcloud


def main():
    df = pd.read_csv('youtoxic_english_1000.csv')
    plot_frequency_of_words(df, 'IsRacist')
    plot_frequency_of_words(df, 'IsReligiousHate')


def plot_frequency_of_words(df: pd.DataFrame, column_name: str):
    comments = df[df[column_name]]
    texts = comments_preprocessing(comments['Text'])
    words = lemmatize_comments(texts)
    plot_word_frequency(words)
    plot_wordcloud(words)


def comments_preprocessing(comments: pd.Series) -> list[list[str]]:
    """Returns list of words for each comment."""
    comments = comments.values.tolist()
    lowercase_comments = lower_comments(comments)
    tokenized_texts = tokenize_comments(lowercase_comments)
    texts = remove_junk_from_comments(tokenized_texts)
    return texts


def lower_comments(comments: list[str]) -> list[str]:
    lowercase_comments = []
    for comment in comments:
        lowercase_comments.append(comment.lower())
    return lowercase_comments


def tokenize_comments(comments: list[str]) -> list[list[str]]:
    tokenized_comments = []
    for comment in comments:
        tokenized_comments.append(word_tokenize(comment))
    return tokenized_comments


def remove_junk_from_comments(comments: list[list[str]]) -> list[list[str]]:
    """comments must be tokenized.
    Junk is stop words and words without alphabetical characters.
    """
    filtered_comments = []
    for comment in comments:
        comment = remove_stop_words(comment)
        comment = remove_non_alphabetical_words(comment)
        filtered_comments.append(comment)
    return filtered_comments


def remove_stop_words(words: list[str]) -> list[str]:
    filtered_words = []
    for word in words:
        if word not in stopwords.words('english'):
            filtered_words.append(word)
    return filtered_words


def remove_non_alphabetical_words(words: list[str]) -> list[str]:
    """Remove all words containing only non-alphabetical characters."""
    filtered_words = []
    for word in words:
        if re.search(r'[a-zA-Z]', word):
            filtered_words.append(word)
    return filtered_words


def lemmatize_comments(comments: list[list[str]]) -> list[str]:
    """Returns list of words."""
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for comment in comments:
        for word in comment:
            lemmatized_words.append(lemmatizer.lemmatize(word))
    return lemmatized_words


if __name__ == '__main__':
    main()

"""Text pre-processing functions"""
import nltk
import string

# Define the tokenizer
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Define the stop-words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Define the stemmer and lemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def cleanup_string(initial_string):
    """Cleanup a document given a single string with its contents

    Parameters
    ----------
    initial_string : string
        string to cleanup

    Returns
    -------
    list(str)
        list of individual words
    """

    # OPTIONAL: Remove obvious stuff using a tokenizer of choice
    tokens = tknzr.tokenize(initial_string)

    return cleanup_wordlist(tokens)


def cleanup_wordlist(words):
    """Cleanup a document provided as al ist of words

    Parameters
    ----------
    words : list(str)
        list of individual words to preprocess

    Returns
    -------
    list(str)
        list of individual words after pre-processing
    """

    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    words = [w.translate(translator) for w in words]

    # Make lower-case
    words = [w.lower() for w in words]

    # Remove english stop-words
    # words = [w for w in words if w not in stop_words]

    # Remove links
    def link(x):
        if ("http" in x) or ("www" in x):
            return "WEBSITE_TOKEN"
        else:
            return x
    words = [link(w) for w in words]

    # Remove numbers
    def number(x):
        if any(char.isdigit() for char in x) or ("½" in x):
            return "NUMBER_TOKEN"
        else:
            return x
    words = [number(w) for w in words]

    # Stemming / Lemmatizing
    # words = [stemmer.stem(w) for w in words]
    words = [lemmatizer.lemmatize(w) for w in words]

    # Remove empty elements, this is equivalent to
    # (item for item in iterable if item)
    words = list(filter(None, words))
    return words


if (__name__ == "__main__"):
    test = "Layin n bed with a headache  ughhhh...waitin on your call..."
    print(cleanup_string(test))

import re
import unicodedata
import nltk
import string
import ssl
import pandas as pd
# import pandas_read_xml as pdx
from nltk.corpus import nps_chat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import trigrams


# from xml.dom import minidom

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('nps_chat')

def main():
    # load text
    with open('corpora/en_US.blogs.txt', 'r') as file:
        for line in file:
            text = line.split()
    print(text[:41])

# Prepare Data
    words = filter(str(text))
    print("After filtering:\n", words)
    print("After cleaning:\n", clean(words))

# Train model
    # get 3-grams
    # assign MLE
    # do katz back-off algorithm


"""
    Normalize text, remove unnecessary characters, 
    perform regex parsing, and make lowercase
"""
def filter(text):
    # normalize text
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    # replace html chars with ' '
    text = re.sub('<.*?>', ' ', text)
    # remove punctuation
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    # only alphabets and numerics
    text = re.sub('[^a-zA-Z]', ' ', text)
    # replace newline with space
    text = re.sub("\n", " ", text)
    # lower case
    text = text.lower()
    # split and join the words
    text = ' '.join(text.split())

    return text

"""
    Remove stopwords and profanity, tokenize remaining words,
    perform lemmatization and POS tagging
"""
def clean(text):
    # words to omit
    stopwords = nltk.corpus.stopwords.words('english')
    bannedwords = open('corpora/bannedwords.txt', "r")

    tokens = nltk.word_tokenize(text)
    wnl = nltk.stem.WordNetLemmatizer()

    output = []
    for words in tokens:
        if words not in stopwords and bannedwords:
            # lemmatize words
            output.append(wnl.lemmatize(words))
    # tag parts of speech
    tokens_tag = nltk.pos_tag(output)

    return tokens_tag


main()

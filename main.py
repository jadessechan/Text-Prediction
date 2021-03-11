import re
import unicodedata
import string
import random
import nltk
from nltk.probability import ConditionalFreqDist


def main():
    file = open('corpora/alice.txt', 'r')
    text = ""
    while True:
        line = file.readline()
        text += line
        if not line:
            break

    # pre-process text
    print("Filtering...")
    words = filter(text)
    print("Cleaning...")
    words = clean(words)

    # make language model
    print("Making model...")
    model = n_gram_model(words)

    print("Enter a phrase: ")
    user_input = input()
    predict(model, user_input)


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
    Tokenize remaining words
    and perform lemmatization
"""
def clean(text):
    tokens = nltk.word_tokenize(text)
    wnl = nltk.stem.WordNetLemmatizer()

    output = []
    for words in tokens:
        # lemmatize words
        output.append(wnl.lemmatize(words))

    return output


"""
    Make a language model using a dictionary, trigrams, 
    and calculate word probabilities
"""
def n_gram_model(text):
    trigrams = list(nltk.ngrams(text, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    # bigrams = list(nltk.ngrams(text, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

# N-gram Statistics
    # get freq dist of trigrams
    # freq_tri = nltk.FreqDist(trigrams)
    # freq_bi = nltk.FreqDist(bigrams)
    # freq_tri.plot(30, cumulative=False)
    # print("Most common trigrams: ", freq_tri.most_common(5))
    # print("Most common bigrams: ", freq_bi.most_common(5))

    # make conditional frequencies dictionary
    cfdist = ConditionalFreqDist()
    for w1, w2, w3 in trigrams:
        cfdist[(w1, w2)][w3] += 1

    # transform frequencies to probabilities
    for w1_w2 in cfdist:
        total_count = float(sum(cfdist[w1_w2].values()))
        for w3 in cfdist[w1_w2]:
            cfdist[w1_w2][w3] /= total_count

    return cfdist

"""
    Generate predictions from the Conditional Frequency Distribution
    dictionary (param: model), append weighted random choice to user's phrase,
    allow option to generate more words following the prediction
"""
def predict(model, user_input):
    user_input = filter(user_input)
    user_input = user_input.split()

    w1 = len(user_input) - 2
    w2 = len(user_input)
    prev_words = user_input[w1:w2]

    # display prediction from highest to lowest maximum likelihood
    prediction = sorted(dict(model[prev_words[0], prev_words[1]]), key=lambda x: dict(model[prev_words[0], prev_words[1]])[x], reverse=True)
    print("Trigram model predictions: ", prediction)

    word = []
    weight = []
    for key, prob in dict(model[prev_words[0], prev_words[1]]).items():
        word.append(key)
        weight.append(prob)
    # pick from a weighted random probability of predictions
    next_word = random.choices(word, weights=weight, k=1)
    # add predicted word to user input
    user_input.append(next_word[0])
    print(' '.join(user_input))

    ask = input("Do you want to generate another word? (type 'y' for yes or 'n' for no): ")
    if ask.lower() == 'y':
        predict(model, str(user_input))
    elif ask.lower() == 'n':
        print("done")
        

main()

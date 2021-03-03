# Text-Prediction

This program implements a statistical trigram language model with NLTK for text prediction based on the Alice in Wonderland corpus.

## Getting started
1. clone or download this repository
```sh
git clone https://github.com/jadessechan/Text-Prediction.git
```
2. run main.py
3. once prompted by the program, enter a phrase related to the corpus

## Demo
Lines 80-86 display n-gram statistics of the corpus and are commented-out by default.

Here is a frequency distribution plot of the most common 30 trigrams:
![frequency distribution of the top 30 trigrams](https://github.com/jadessechan/Text-Prediction/blob/master/images/trigram_fdplot.png)

Here is an example of the program output:
![demo image of running program](https://github.com/jadessechan/Text-Prediction/blob/master/images/demo.png)
### final output of demo:
User input: ***alice said to the*** <br />
Prediction: ***alice said to the table, half hoping she might find another***
*(comma was added for readability)* <br />
What did alice want to find again?? The suspense...😖 <br />

## Implementation
I used NLTK's probability library to store the probability of each predicted word,
```sh
ConditionalFreqDist()
```
then the program picks from a weighted random probability to decide which prediction to append to the given phrase.
```sh
random.choices()
```
The user decides when to stop the program by choosing whether or not to predict the next word.
```sh
"Do you want to generate another word? (type 'y' for yes or 'n' for no): "
```

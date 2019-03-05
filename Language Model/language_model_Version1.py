from collections import *
from random import random
import operator
# from numpy import product
from math import exp, log


def train_char_lm(fname, order, add_k=1):
    ''' Trains a language model.
    This code was borrowed from
    http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

    Inputs:
        fname: Path to a text corpus.
        order: The length of the n-grams.
        add_k: k value for add-k smoothing.

    Returns:
        A dictionary mapping from n-grams of length n to a list of tuples.
        Each tuple consists of a possible net character and its probability.
    '''

    # opening the file, reading it, adding a padding
    data = open(fname, encoding='latin-1').read()
    # calling the defaultdict class and inheriting the Counter class
    lm = defaultdict(Counter)

    pad = "~" * order
    data = pad + data

    # the padding is required. otherwise the first n letters of the corpus will not have probabilities for them in those
    # positions i.e. First Citizen, without padding will not collect probabilities for F or i at the start of a word. to
    # address this, the padding has to be the same as the n in the n-gram. simply appending the padding to the start of
    # the corpus we retain the original code for lm training if add_K ==0, i.e. no smoothing because this is the most
    # space efficient data structure for this purpose. it also doesn't affect our code written for perplexity() below

    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        # char is the letter we are collecting the count (+=1) for the conditional probability. history is the
        # preceding n-letters before char. by using the defaultdict class with history as the key, we are storing the
        # count for each n-gram. e.g. r,v,n, for Fi... (for Fi[r]|st, Fi[v]|e, Fi[n]|e etc]
        lm[history][char] += 1

    vocabulary = set(data) # set of all characters = Vocabulary
    # add_k - populate the dictionary with add_k values for
    for history in lm.keys():
        for char in vocabulary:
            lm[history][char] += add_k

    # support for unseen history
    # The probability for every character from any unseen context with the add-k method is the same
    # Here we use the None context as a default value
    for char in vocabulary:
        lm[None][char] = add_k

    # Normalisation step
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c, cnt/s) for c, cnt in counter.items()]
    outlm = {hist: normalize(chars) for hist, chars in lm.items()}
    return outlm


def generate_letter(lm, history, order):
    ''' Randomly chooses the next letter using the language model.
    Inputs:
        lm: The output from calling train_char_lm.
        history: A sequence of text at least 'order' long.
        order: The length of the n-grams in the language model.

    Returns:
        A letter
    '''
    history = history[-order:]
    dist = lm[history]
    x = random()
    for c, v in dist:
        x = x - v
        if x <= 0:
            return c


def generate_text(lm, order, nletters=500):
    '''
    Generates a bunch of random text based on the language model.

    Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of previous text.
    order: The length of the n-grams in the language model.

    Returns:
    A letter
    '''
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)


def probability(history, character, lm, order):
    """
    Extracts the probability of given character depending on given history.
    Inputs:
        char: Character to compute the probability of.
        history: A sequence of previous text.
        lm: The output from calling train_char_lm.
        order: The length of the n-grams in the language model.
    Output:
        Float value for probability
    """

    # access the distributions corresponding to history:
    history = history[-order:]

    # support for unseen history
    try:
        distribution = lm[history]
    except KeyError:
        distribution = lm[None]
        
    # look for the character to predict in the distribution
    for char, val in distribution:
        if char == character:
            return val

    # if the character was not found, raise an error
    raise ValueError


def perplexity(test_filename, lm, order):
    '''
    Computes the perplexity of a text file given the language model.
    Inputs:
        test_filename: path to text file
        lm: The output from calling train_char_lm.
        order: The length of the n-grams in the language model.
    Output:
        Float value for perplexity
    '''
    test = open(test_filename, encoding="latin-1").read()
    pad = "~" * order
    test = pad + test

    try:
        # Use the sum of logarithms instead of products
        total_prob = exp(
            sum(
                log(probability(test[:index], test[index], lm, order))
                for index in range(order, len(test)-order)
        ))

        perplexity = total_prob ** (-1 / (len(test) - order))
        return perplexity

    except ValueError:
        # A value error is raised if an unknown character is encountered
        return float("inf")


def build_lm(fname, order=4, add_k=1):
    lms = []
    for i in range(order):
        lm = train_char_lm(fname, order-i, add_k)
        lms.append(lm)
    return lms


def calculate_prob_with_backoff(char, history, lms, lambdas):
    '''Uses interpolation to compute the probability of char given a series of
    language models trained with different length n-grams.
    Inputs:
    char: Character to compute the probability of.
    history: A sequence of previous text.
    lms: A list of language models, outputted by calling train_char_lm.
    lambdas: A list of weights for each lambda model. These should sum to 1.
    Returns:
    Probability of char appearing next in the sequence.
    '''

    # Get the probabilities from every language model
    probabilities = [probability(history, char, lms[i], order-i) for i in range(len(lms))]
    
    # Compute the weighted sum of the probabilities from every language model
    prob = sum(probabilities[i] * lambdas[i] for i in range(len(probabilities)))
    return prob


def set_lambdas(lms, dev_filename=None):
    '''Returns a list of lambda values that weight the contribution of each n-gram model
    This can either be done heuristically or by using a development set.

    Currently randomly generating lambdas

    Inputs:
    lms: A list of language models, outputted by calling train_char_lm.
    dev_filename: Path to a development text file to optionally use for tuning the lmabdas.
    Returns:
    a list of lambda values.
    '''
    # generating random lambdas
    lambdas = [random() for i in range(len(lms))]

    # Normalisation of the lambdas
    lambda_sum = sum(lambdas)
    lambdas = [lamb/lambda_sum for lamb in lambdas]
    return lambdas


if __name__ == '__main__':
    print('Training language model')
    order = 4
    add_k = 1

    #lm = train_char_lm("shakespeare_input.txt", order, add_k)
    #print(perplexity("shakespeare_sonnets.txt", lm, order))
    #print(perplexity("nytimes_article.txt", lm, order))
    
    lms = build_lm("shakespeare_input.txt", order, add_k)
    # lambdas = set_lambdas(lms)
    lambdas = [0.4, 0.3, 0.2, 0.1]

    print(calculate_prob_with_backoff("e", "henc", lms, lambdas))
    lambdas = [0.1, 0.2, 0.3, 0.4]
    print(calculate_prob_with_backoff("e", "henc", lms, lambdas))

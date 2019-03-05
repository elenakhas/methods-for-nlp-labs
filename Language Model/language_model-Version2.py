from collections import *
from random import random
import numpy as np
import math

def train_char_lm(fname, order=4, add_k=1):
    ''' Trains a language model.

    This code was borrowed from
    http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

    Inputs:
    fname: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED

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

    if add_k == 0:

        for i in range(len(data)-order):
            # len(data)-order is to contain the iteration prevent "OutofRange" error
            history, char = data[i:i+order], data[i+order]
            # char is the letter we are collecting the count (+=1) for the conditional probability. history is the
            # preceding n-letters before char. by using the defaultdict class with history as the key, we are storing the
            # count for each n-gram. e.g. r,v,n, for Fi... (for Fi[r]|st, Fi[v]|e, Fi[n]|e etc]

            lm[history][char]+=1

        def normalize(counter):
            # this function is to 1. go to each history
            # and transform the count for each of the char within it into
            # probabilities wrt to all of the other chars within that history,

            s = float(sum(counter.values()))
            # placing the "normalised" results for chars in a list
            return [(c,cnt/s) for c,cnt in counter.items()]


    else:
        # adding smoothing
        # 1. let's get every character found within the corpus - we can use the set() method

        chars_corpus_wide = list(set(data))

        # 2. let's build out lm, adding every history to it. within it, +=1 for each char follows each history
        for i in range(len(data)-order):
            history, char = data[i:i+order], data[i+order]
            lm[history][char]+=1

        # 3. we iterate through all the keys fof the lm (i.e. each "vocab" in the corpus)
        # for each character (a,b,c, /n etc) found within the corpus, we add the value of
        # add_k to it. i.e. laplace smoothing, add_k = 1. This is the default we have set above.
        for j2 in lm.keys():
            for j3 in range(len(chars_corpus_wide)):
                char_corpus_wide = chars_corpus_wide[j3]
                lm[j2][char_corpus_wide] += add_k

        def normalize(counter):
            s = float(sum(counter.values()))
            # we add the number of characters * add_k within the corpus to the denominator (i.e. Count(W1...,Wi_minus1))
            # read section 3.4.2 of Jurafsky
            s_smoothing = s + add_k*float(len(lm))
            return [(c,cnt/s_smoothing) for c,cnt in counter.items()]


    # storing the "normalised" char values in a dictionary with the
    # structure as the defaultdict from before
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}

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
    for c,v in dist:
        x = x - v
        if x <= 0: return c


def generate_text(lm, order, nletters=500):
    '''Generates a bunch of random text based on the language model.

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

def perplexity(test_filename, lm, order):
    '''Computes the perplexity of a text file given the language model.

    Inputs:
        test_filename: path to text file
        lm: The output from calling train_char_lm.
        order: The length of the n-grams in the language model.
    '''
    test = open(test_filename, encoding='latin-1').read()
    pad = "~" * order
    test = pad + test

    # TODO: YOUR CODE HRE
    cond_prob_list = []
    for i in range(len(test)-order):
        # from the order-n letter take the order-n preceding letters
        # and go to the lm with those order-n letters as the key.
        # extract the letter and take its probability.

        # Wi  starts from the order-n+1 character in the file (after the order-n paddings)
        char = test[i+order]
        hist = test[0+i:order+i]

        # iterate through the length of lm[hist]
        # where lm[hist] is a dictionary whose keys are every possible char that follows
        # hist. e.g. r in First, v in Five etc..., we take the probability for char to follow hist.
        try:
            for i2 in range(len(lm[hist])):
                if lm[hist][i2][0]==char:
                    cond_prob_list.append(lm[hist][i2][1])

        except:
            interpolation_prob = calculate_prob_with_backoff(char, hist, lms, lambdas)
            # initially I took probability to be 0 if a particular hist in the test data does not appear in the lm,
            # or if there is no entry for char within lm[hist]... this was causing a problem with computing perplexity
            # with interpolation implemented through calculate_prob_with_backoff(), this was partially solved,
            # the lambdas used were randomly generated. per Jurafsky, an EM algorithm should be implemented to
            # find the optimal lambda in order to provide the maximum probability for the test/held-out set.
            if interpolation_prob>0: cond_prob_list.append(interpolation_prob)
            else: cond_prob_list.append(math.inf)

    # to avoid underflow, we use log probabilities in computing perplexity using the standard formula.
    # in doing so:
    # 1. we can also make use of the property of log functions. log(ab) = log(a) + log(b)
    # 2. we have to use the log likelihood form of the perplexity formula
    # see page 22 of https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    # and page 3 of http://ssli.ee.washington.edu/WS07/notes/ngrams.pdf

    log_prob_sum = sum([np.log2(i) for i in cond_prob_list])
    # in a word-based lm, we divide the sum of the log probs by the number of words (not type) within the data
    # since we are working with characters here, we divide by the number of chars within the
    Nwords_in_test = len(test)
    h_entropy = (-1/Nwords_in_test)*log_prob_sum
    perplexity = 2**h_entropy

    return perplexity

def calculate_prob_with_backoff(char, history, lms, lambdas):
    '''Uses interpolation to compute the probability of char given a series of
     language models trained with different length n-grams.

    Inputs:
     char: Character to compute the probability of.
     history: A sequence of previous text.
     lms: A list of language models, outputted by calling train_char_lm. ordered from
     smallest ngram size to largest
     lambdas: A list of weights for each lambda model. These should sum to 1.

    Returns:
    Probability of char appearing next in the sequence.
    '''
    # TODO: YOUR CODE HRE
    prob_char_intp_list = []
    # iterate thorugh the list of lms
    for lm in lms:
        # get the size of the ngram from that particular lm
        size_gram = len(list(lm.keys())[0])
        # set a new history based on the size of the ngrams of that particular lm
        new_hist = history[-size_gram:]
        # use try except to handle cases where P(char|new_history) is not in a particular lm (i.e. OOV)
        try:
            for c,v in lm[new_hist]:
                if c == char:
                    prob_char_intp_list.append(lambdas[size_gram-1]*v)
        except:
            prob_char_intp_list.append(0)

    return float(sum(prob_char_intp_list))


def set_lambdas(lms, dev_filename=None):
    '''Returns a list of lambda values that weight the contribution of each n-gram model

    This can either be done heuristically or by using a development set.

    Inputs:
    lms: A list of language models, outputted by calling train_char_lm.
    dev_filename: Path to a development text file to optionally use for tuning the lmabdas.

    Returns:
    Probability of char appearing next in the sequence.
    '''
    # there seems to be a mistake in the docstring above. it should read: "Returns: list of lambda values"

    # TODO: YOUR CODE HERE
    # generate a list of random numbers with the same number of elements as lms [i.e. 1 lambda for each lms model to be
    # used in computing interpolation]
    lambdas = [random() for i in range(len(lms))]
    # normalise so that lambda elements sum to 1
    lambdas = [i/sum(lambdas) for i in lambdas]
    return lambdas

if __name__ == '__main__':
    print('Training language model')
    lm_order = 4
    fname = "/Users/k1000mbp/Desktop/0_MSc_NLP_Nancy/01_Semester_1/09_Methods_for_NLP/Assignment3/shakespeare_input.txt"
    lm = train_char_lm(fname, order = lm_order)

    # the resulting lms will have models from 1-gram to lm_order-gram. ordered in increasing gram size
    # e.g. [1-gram lm, 2-gram lm, .... lm_order-gram lm]
    lms = [train_char_lm(fname, order=i) for i in range(1,lm_order+1)]
    lambdas = set_lambdas(lms)
    print (lambdas)
    print("Test calculate_prob_with_backoff function: backoff probability of \"e\" appearing after \"fing\": ", calculate_prob_with_backoff("e", "fing", lms, lambdas))

    test_filename1 = "/Users/k1000mbp/Desktop/0_MSc_NLP_Nancy/01_Semester_1/09_Methods_for_NLP/Assignment3/test_data/nytimes_article.txt"
    test_filename2= "/Users/k1000mbp/Desktop/0_MSc_NLP_Nancy/01_Semester_1/09_Methods_for_NLP/Assignment3/test_data/shakespeare_sonnets.txt"
    test_files = [test_filename1, test_filename2]
    for file in test_files:
        print("Perplexity of {}".format(file.split("/")[-1]), perplexity(file, lm, order = lm_order))

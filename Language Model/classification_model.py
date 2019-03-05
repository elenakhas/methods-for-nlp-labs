from collections import *
from random import random
import numpy as np
import math
import operator


def train_char_lm(fname, country_code, order=4, add_k=1, lm=None):
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
    data = open(fname.format(country_code), encoding='latin-1').read()
    # calling the defaultdict class and inheriting the Counter class
    if lm == None:
        lm = defaultdict(Counter)

    pad = "~" * order
    data = pad + data
    # some data pre-processing. besides padding the start of the file, we want to pad the start of each city name
    # we use the .replace() function to change newlines in the file to the padding. since the files are all lowercase,
    # we don't have to deal with that.
    data = data.replace("\n", "~" * order)

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

            lm[history][country_code]+=1

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


        # 2. let's build out lm, adding every history to it. within it, +=1 for each char follows each history
        for i in range(len(data)-order):
            history, char = data[i:i+order], data[i+order]
            lm[history][country_code]+=add_k

    return lm


# write a normalize() function to return probabilities from counts.
def normalize(counter, add_k=1):
    s = float(sum(counter.values()))
    # we add the number of characters * add_k within the corpus to the denominator (i.e. Count(W1...,Wi_minus1))
    # read section 3.4.2 of Jurafsky
    s_smoothing = s + add_k*float(len(lm))
    return [(c,cnt/s_smoothing) for c,cnt in counter.items()]


# write a function to obtain the probability of a city being in that country.
def get_prob_score(cityname, lm, order, country_list):
    '''
    given a returns a tuple with the cityname, the countryname, and the city's probability of being from this country
    input:
    cityname: the city to search for
    country_code:
    lms : a list of lm-s, each trained on the names of a different country's cities
    order: the size of the history of the lm-s
    '''
    prob_score_dict = {}

    for country in country_list:
        prob_score_dict[country] = 0

    for i in range(len(cityname)-order):
        hist = cityname[i:i+order]
        try:
            for c, v in lm[hist]:
                prob_score_dict[c] += v

        except:
            for i in fname_list:
                prob_score_dict[c] += 0

    return prob_score_dict


if __name__ == '__main__':

    fname_list = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]
    fname= '/Users/k1000mbp/Desktop/0_MSc_NLP_Nancy/01_Semester_1/09_Methods_for_NLP/Assignment3/train/{}.txt'
    lm_order = 4

    # start with lm as None
    lm = None
    # iterate through the fname_list and increment
    for i in fname_list:
        lm = train_char_lm(fname, i, order=lm_order, lm=lm)

    # apply the normalise function to return the probabilities from the count
    lm = {hist:normalize(chars) for hist, chars in lm.items()}

    # open the test file
    test_fname= '/Users/k1000mbp/Desktop/0_MSc_NLP_Nancy/01_Semester_1/09_Methods_for_NLP/Assignment3/cities_test.txt'
    test_data = open(test_fname, encoding='latin-1').read().split("\n")

    # let's get some results
    results = []
    for cityname in test_data:
        cityname_padded = "~"*lm_order+cityname

        prob_scores = get_prob_score(cityname_padded, lm, lm_order, fname_list)
        most_prob = max(prob_scores.items(), key=operator.itemgetter(1))[0]

        results.append("".join([cityname.lstrip("~"*lm_order), '\t', most_prob, "\n"]))

    # write the results to a txt file
    print("".join(results))
    results_fname = '/Users/k1000mbp/Desktop/0_MSc_NLP_Nancy/01_Semester_1/09_Methods_for_NLP/Assignment3/labels.txt'
    results_write = open(results_fname, "w+")

    for i in results:
        results_write.write(i)

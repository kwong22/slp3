#!/usr/bin/python

import numpy as np
import pandas as pd
import sys

"""
Usage:
    python nb_model.py train_file k binary
        Trains a naive Bayes classifier and prints log likelihoods and log
        priors.
    python nb_model.py train_file k binary test_file
        Trains a naive Bayes classifier, prints log likelihoods and log priors,
        and uses the classifier to predict labels on test examples.

    train_file: name of file containing one training example per line
        Each example should contain space-delimited tokens.
        All but the last token are taken to be the input.
        The last token is taken to be the gold label.
    k: for add-k smoothing, the number to add to all counts of words
    binary: int specifying binary naive Bayes classifier or not
        train a binary multinomial NB classifier if binary is 1
        train a multinomial NB classifier otherwise
    test_file: name of file containing one test example per line
        Each example should contain space-delimited tokens.
        All tokens are taken as the input.
"""

def tokenize(filename, keepcase=True, vocab=None):
    """
    Split file line-by-line into tokens, leaving out unknown tokens.

    Inputs:
    - filename: name of the file to read from
    - keepcase: True to keep case of letters
        False to convert all letters to uppercase
    - vocab: list of known words
        if vocab is not None, then words not in vocab are excluded from output.
        Otherwise, all tokens are added to the output.

    Returns:
    - output: list of lists of tokens from the file
    """
    output = []

    with open(filename, 'r') as lines:
        for line in lines:
            # Remove newlines and punctuation
            line = line.replace('\n', '').replace(',', '').replace('.', '')

            if not keepcase:
                line = line.upper()

            line = line.split(' ')

            tokens = []

            # If vocab is provided, only include known tokens
            if vocab is not None:
                for token in line:
                    if token in vocab:
                        tokens.append(token)
            else:
                # If no vocab is provided, include all tokens
                tokens = line

            output.append(tokens)

    return output

def build_examples(data):
    """
    Build examples (input, class) from a list of lists of tokens.
    The class is assumed to be the last token in each list. The rest of the
    list is input.

    Inputs:
    - data: list of lists of tokens

    Returns a tuple of:
    - inputs: list of lists of input words for each example
    - classes: list of classes for the examples
    """
    inputs = []
    classes = []

    for i in range(len(data)):
        inputs.append(data[i][:-1]) # input includes all but last word
        classes.append(data[i][-1]) # class is last word

    return inputs, classes

def flatten_list(X):
    """
    Flatten list of lists.

    Inputs:
    - X: list of lists

    Returns:
    - a flattened list of elements in x
    """
    return [el for x in X for el in x]

def train_nb_model(X, y, k=0, binary=False):
    """
    Train a naive Bayes classifier using add-k smoothing.

    Inputs:
    - X: list of lists of input words for each example
    - y: list of classes for the examples
    - k: number to add to all counts
    - binary: if True, trains a binary NB classifier (up to 1 count per word per
      document)

    Returns a tuple of:
    - vocab: list of unique word types that make up the input
    - classes: list of possible classes
    - loglikelihoods: log likelihoods of each word given classes, shape (V, C)
        V is number of words in the vocabulary
        C is number of classes
    - logpriors: log prior probabilities of the classes, shape (C,)
    """
    N = len(X)

    # Get unique words in the inputs
    vocab = np.unique(flatten_list(X))
    V = len(vocab)

    # Get the set of possible classes
    classes, class_counts = np.unique(y, return_counts=True)
    C = len(classes)

    # Calculate the log prior probabilities of the classes
    logpriors = np.log(class_counts / np.sum(class_counts))

    # Count occurrences of words in the inputs
    counts = np.zeros((N, V))
    for i in range(N):
        x_vals, x_counts = np.unique(X[i], return_counts=True)
        for j in range(len(x_vals)):
            ind = np.where(vocab == x_vals[j])
            counts[i, ind] = x_counts[j]

    # Clip counts to 1 if binary NB classifier
    if binary == True:
        counts = np.clip(counts, a_min=None, a_max=1)

    # Add-k smoothing
    counts += k

    # Calculate the log likelihoods of each word given the classes
    loglikelihoods = np.zeros((V, C))
    for c in range(C):
        # Get subset of counts for documents of current class
        class_table = counts[np.where(np.array(y) == classes[c])[0]]

        probs = np.sum(class_table, axis=0) / np.sum(class_table)
        loglikelihoods[:, c] = np.log(probs)

    return vocab, classes, loglikelihoods, logpriors

def print_table(row_names, col_names, entries):
    """
    Print table with row and column names.

    Inputs:
    - row_names: list of names of rows
    - col_names: list of names of columns
    - entries: 2-D Numpy array containing entries

    Prints table.
    """
    df = pd.DataFrame(entries, index=row_names, columns=col_names)
    print(df)

def test_nb_model(train_file, k, binary, test_file=None):
    """
    Test naive Bayes classifier.

    Inputs:
    - train_file: file containing training data
    - k: number to add to all N-gram counts
    - binary: int specifying binary NB classifier or not
        1 indicates binary (clip counts of words in each document to 1)
        otherwise not binary
    - test_file: file containing testing data
    """
    tokens = tokenize(sys.argv[1])
    X, y = build_examples(tokens)
    vocab, classes, loglikelihoods, logpriors = train_nb_model(X, y, k=k,
            binary=(binary == 1))

    print('\nLog likelihoods:')
    print_table(vocab, classes, loglikelihoods)
    print('\nLog priors:')
    print_table(['logP(c)'], classes, logpriors[np.newaxis, :])

    # Classify test documents if provided
    if test_file is not None:
        X_test = tokenize(test_file, vocab=vocab)
        preds = classify_text(X_test, vocab, classes, loglikelihoods, logpriors)

        print('\nPredictions:')
        for i in range(len(X_test)):
            print(X_test[i], '->', preds[i])

def classify_text(X_test, vocab, classes, loglikelihoods, logpriors):
    """
    Classify a set of documents using likelihoods and prior probabilities.

    Inputs:
    - X_test: list of lists of input words for each test example
    - vocab: list of word types that make up the input
    - classes: list of possible classes
    - loglikelihoods: log likelihoods of each word given classes, shape (V, C)
        V is number of words in the vocabulary
        C is number of classes
    - logpriors: log prior probabilities of the classes, shape (C,)

    Returns:
    - preds: list of the classes with highest probability given the documents
    """
    preds = []

    for x_test in X_test:
        logprobs = []
        for c in range(len(classes)):
            # Calculate log probability of each document given the class
            # (leaving out P(document) from denominator)
            # log probability = log prior + sum(log likelihoods)
            logprob = logpriors[c]
            for token in x_test:
                ind = np.where(vocab == token)
                logprob += loglikelihoods[ind, c]
            logprobs.append(logprob)

        # Choose class with highest probability
        pred = np.argmax(logprobs)
        preds.append(classes[pred])

    return preds

if __name__ == '__main__':
    if len(sys.argv) == 4:
        test_nb_model(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 5:
        test_nb_model(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]),
                sys.argv[4])
    else:
        raise ValueError('Invalid number of arguments: %d' % len(sys.argv))

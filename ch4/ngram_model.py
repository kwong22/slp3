#!/usr/bin/python

import numpy as np
import pandas as pd
import sys

def tokenize(filename, keepcase=True, N=1):
    """
    Split file into tokens, adding sentence boundaries as necessary.

    Inputs:
    - filename: name of the file to read from
    - keepcase: True to keep case of letters
        False to convert all letters to uppercase
    - N: number of words to include in N-grams, at least 1

    Returns:
    - output: list of tokens from the file
    """
    stream = open(filename).read()
    
    tokens = stream.replace('\n', ' ').split(' ')
    tokens = list(filter(None, tokens)) # remove empty strings

    if not keepcase:
        tokens = tokens.upper()

    output = []

    start_symbol = '<s>'
    end_symbol = '</s>'

    # Add sentence boundaries
    for token in tokens:
        output.append(token)
        end_punctuation = ['.', '?', '!']
        if token in end_punctuation:
            # Add end-symbol to end of sentence
            output.append(end_symbol)

            # Add N-1 start-symbols to beginning of sentence
            for _ in range(N-1):
                output.append(start_symbol)

    return output

def build_ngrams(corpus, N=1):
    """
    Build N-grams from a corpus.

    Inputs:
    - corpus: list of words
    - N: number of words to include in N-grams, at least 0
        if N is 0, returns list containing an empty string

    Returns:
    - ngrams: list of N-grams, joined internally with spaces
    """
    if N == 0:
        return ['']

    if N == 1:
        return corpus

    if N > len(corpus):
        raise ValueError('Not enough words in corpus to create N-gram')

    ngrams = []

    for i in range(len(corpus)-N+1):
        ngrams.append(' '.join(corpus[i:i+N]))

    return ngrams

def count_ngrams(ngrams):
    """
    Count N-grams.

    Inputs:
    - ngrams: list of N-grams

    Returns a tuple of:
    - values: list of unique N-grams
    - counts: list of counts of the N-grams
    """
    return np.unique(ngrams, return_counts=True)

def train_ngram_model(corpus, N=1, k=0):
    """
    Train an N-gram language model using add-k smoothing.

    Inputs:
    - corpus: list of words
    - N: number of words to include in N-grams, at least 1
    - k: number to add to all counts

    Returns:
    - probs: probability estimations for N-grams
    """
    mgrams = build_ngrams(corpus, N=N-1) # (N-1)-grams
    ngrams = build_ngrams(corpus, N=N) # N-grams

    # Get unique values and counts for the N-grams and (N-1)-grams
    m_values, m_counts = count_ngrams(mgrams)
    n_values, n_counts = count_ngrams(ngrams)

    # Get unique last words in N-grams for predicting from all (N-1)-grams
    next_words = np.unique([ngram.split(' ')[-1] for ngram in n_values])

    probs = np.zeros((len(m_values), len(next_words)))

    # Place counts into matrix
    for i in range(len(m_values)):
        for j in range(len(next_words)):
            next_sequence = ' '.join((m_values[i], next_words[j])).strip()
            if next_sequence in n_values:
                probs[i, j] = n_counts[np.where(n_values == next_sequence)]

    # Calculate probability estimates
    probs += k # add-k smoothing
    probs /= np.sum(probs, axis=1, keepdims=True)

    return m_values, next_words, probs

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

def test_ngram_model(corpus_file, N, k):
    """
    Test N-gram model.

    Inputs:
    - corpus_file: file containing corpus
    - N: number of words to include in N-grams
    - k: number to add to all N-gram counts
    """
    tokens = tokenize(corpus_file)
    m_values, next_words, probs = train_ngram_model(tokens, N=N, k=k)
    print_table(m_values, next_words, probs)

def generate_sentence(corpus_file, N, k, max_len=50):
    """
    Train N-gram model and generate a sentence.

    Inputs:
    - corpus_file: file containing corpus
    - N: number of words to include in N-grams
    - k: number to add to all N-gram counts

    Returns:
    - string containing words sampled from the model
    """
    tokens = tokenize(corpus_file, N=N)
    prev_grams, next_words, probs = train_ngram_model(tokens, N=N, k=k)

    end_symbol = '</s>'

    # Store sampled words in a list
    words = []

    # Randomly initialize with an (N-1)-gram
    seed_gram = np.random.choice(prev_grams)
    words = seed_gram.split(' ')

    # Sample words until end-symbol is chosen or max sentence length is reached
    num_words = len(words)
    while (words[-1] != end_symbol) & (num_words < max_len):
        # Use previous N-1 words to sample next word
        prev_gram = ' '.join(words[-(N-1):])

        # Get probability distribution if there is one
        # No probability distribution if unigram model
        prob_dist = None
        if prev_gram in prev_grams:
            ind = np.where(prev_grams == prev_gram)
            prob_dist = probs[ind].flatten()

        next_word = np.random.choice(next_words, p=prob_dist)
        words.append(next_word)

        num_words += 1

    return ' '.join(words)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        #test_ngram_model(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
        sentence = generate_sentence(sys.argv[1],
                int(sys.argv[2]), float(sys.argv[3]))
        print(sentence)
    else:
        raise ValueError('Invalid number of arguments: %d' % len(sys.argv))

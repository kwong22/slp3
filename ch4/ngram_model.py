#!/usr/bin/python

from itertools import product
import numpy as np
import pandas as pd
import sys

def tokenize(filename, keepcase=True, N=1, vocab=None):
    """
    Split file into tokens, adding unknown symbols and sentence boundaries as
    necessary.

    Inputs:
    - filename: name of the file to read from
    - keepcase: True to keep case of letters
        False to convert all letters to uppercase
    - N: number of words to include in N-grams, at least 1
    - vocab: list of known words
        if vocab is not None, then unknown-symbols are added to the output in
        place of tokens not in the vocabulary. Otherwise, all tokens are added
        to the output.

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
    unk_symbol = '<UNK>'

    for token in tokens:
        # If vocabulary is given, replace unknown tokens with <UNK>
        if vocab is not None:
            if token in vocab:
                output.append(token)
            else:
                output.append(unk_symbol)
        else:
            output.append(token)

        # Add sentence boundaries
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
    # Get unique words in the corpus
    vocab = np.unique(corpus)

    unk_symbol = '<UNK>'
    vocab = np.append(vocab, unk_symbol)

    # Form all possible (N-1)-grams from the vocabulary
    prods = list(product(vocab, repeat=N-1))
    prev_grams = np.array([' '.join(prod) for prod in prods])

    # Build and count the N-grams in the corpus
    ngrams = build_ngrams(corpus, N=N)
    n_values, n_counts = count_ngrams(ngrams)

    probs = np.zeros((len(prev_grams), len(vocab)))

    # Place counts into corresponding matrix entries
    for i in range(len(prev_grams)):
        for j in range(len(vocab)):
            next_sequence = ' '.join((prev_grams[i], vocab[j])).strip()
            if next_sequence in n_values:
                probs[i, j] = n_counts[np.where(n_values == next_sequence)]

    # Calculate probability estimates
    probs += k # add-k smoothing
    probs /= np.sum(probs, axis=1, keepdims=True)

    return prev_grams, vocab, probs

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
    prev_grams, vocab, probs = train_ngram_model(tokens, N=N, k=k)

    start_symbol = '<s>'
    end_symbol = '</s>'

    # Store sampled words in a list
    words = []

    # Initialize with start-symbols
    for _ in range(N-1):
        words.append(start_symbol)

    # Sample words until end-symbol is chosen or max sentence length is reached
    num_words = len(words)
    while (words[-1] != end_symbol) & (num_words < max_len):
        # Use previous N-1 words to sample next word
        prev_gram = ' '.join(words[-(N-1):])

        # Get probability distribution if there is one
        # No probability distribution if unigram model
        prob_dist = None
        if prev_gram in prev_grams:
            prob_dist = probs[np.where(prev_grams == prev_gram)].flatten()

        next_word = np.random.choice(vocab, p=prob_dist)
        words.append(next_word)

        num_words += 1

    return ' '.join(words)

def compute_perplexity(train_file, N, k, test_file):
    """
    Train N-gram model with a corpus, then compute perplexity of the model on a
    test corpus.

    Inputs:
    - train_file: file containing training corpus
    - N: number of words to include in N-grams
    - k: number to add to all N-gram counts
    - test_file: file containing test corpus

    Returns:
    - perp: perplexity of the model on the test set
    """
    # Train N-gram model
    tokens = tokenize(train_file, N=N)
    prev_grams, vocab, probs = train_ngram_model(tokens, N=N, k=k)

    # Build N-grams from test set
    test_tokens = tokenize(test_file, N=N, vocab=vocab)
    test_ngrams = build_ngrams(test_tokens, N=N)
    num_test = len(test_ngrams)

    test_probs = np.zeros(num_test)

    # Retrieve probabilities for each N-gram in the test set
    for i in range(num_test):
        # Given previous N-1 words, find probability of next word
        test_ngram = test_ngrams[i].split(' ')
        prev_gram = ' '.join(test_ngram[:-1])
        next_word = test_ngram[-1]

        prev_ind = np.where(prev_grams == prev_gram)
        next_ind = np.where(vocab == next_word)
        test_probs[i] = probs[prev_ind, next_ind]

    log_perp = 1 / num_test * np.sum(-np.log(test_probs))
    perp = np.exp(log_perp)

    return perp

if __name__ == '__main__':
    if len(sys.argv) == 4:
        #test_ngram_model(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
        sentence = generate_sentence(sys.argv[1],
                int(sys.argv[2]), float(sys.argv[3]))
        print(sentence)
    elif len(sys.argv) == 5:
        perp = compute_perplexity(sys.argv[1], int(sys.argv[2]),
                float(sys.argv[3]), sys.argv[4])
        print(perp)
    else:
        raise ValueError('Invalid number of arguments: %d' % len(sys.argv))

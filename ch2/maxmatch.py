#!/usr/bin/python

import numpy as np
import sys

from min_edit_dist import *

"""
Usage:
    python maxmatch.py sentences dictionary
        Segments sentences into words using the maximum matching (MaxMatch)
        algorithm. Matches are case-insensitive.
        For each sentence, compares segmentation of original sentence to
        segmentation of sentence with whitespace removed to compute Word Error
        Rate. Prints results.

    sentences: name of file containing one sentence per line
    dictionary: name of file containing dictionary, one word per line
"""

def read_lines_from_file(filename, keepcase=True):
    """
    Build a list of strings line-by-line from a file.

    Inputs:
    - filename: name of the file to read from
    - keepcase: True to keep case of letters
        False to convert all letters to uppercase

    Returns:
    - output: list of strings from the file
    """
    output = []
    with open(filename, 'r') as lines:
        for line in lines:
            line = line.replace('\n', '')

            if not keepcase:
                line = line.upper()

            output.append(line)

    return output

def maxmatch(sentence, dictionary):
    """
    Segment sentence into words using the maximum matching (MaxMatch)
    algorithm. Matches are case-insensitive.

    Inputs:
    - sentence: the sentence to segment into words
    - dictionary: list of uppercase words

    Returns:
    - words: list of words from the sentence
    """
    N = len(sentence)

    if N < 1:
        return []

    words = []

    # Check if first i characters make up a word in the dictionary, down to i=1
    for i in reversed(range(1, N+1)):
        if sentence[0:i].upper() in dictionary:
            words.append(sentence[0:i])
            return words + maxmatch(sentence[i:N], dictionary)

    # No multi-character word found, add single character if not whitespace
    if not sentence[0].isspace():
        words.append(sentence[0])

    return words + maxmatch(sentence[1:N], dictionary)

def test_maxmatch(sentences, dictionary):
    """
    Test the maximum matching algorithm with a set of sentences.
    Evaluate using Word Error Rate.
    WER = # insertions, deletions, and substitutions / # words in gold list

    Inputs:
    - sentences: list of strings to segment
    - dictionary: list of uppercase words
    - mean Word Error Rate for the set of sentences

    Prints results
    """
    N = len(sentences)

    # Use algorithm to create gold lists of words
    gold_lists = [maxmatch(s, dictionary) for s in sentences]

    # Test algorithm with same sentences with whitespace removed
    test_sentences = [s.replace(' ', '') for s in sentences]
    test_lists = [maxmatch(s, dictionary) for s in test_sentences]

    # Calculate Word Error Rate for each sentence compared to gold segmentation
    dists = [minimum_edit_distance(test_lists[i], gold_lists[i]) for i in range(N)]
    wer = [dists[i] / len(gold_lists[i]) for i in range(N)]

    # Print results
    for i in range(N):
        print('\n### Sentence %d ###' % (i+1))
        print('Original sentence:')
        print(sentences[i])
        print('Gold segmentation:')
        print(gold_lists[i])
        print('Test segmentation:')
        print(test_lists[i])
        print('Word error rate: %f' % wer[i])

    print('\nMean WER: %f' % np.mean(wer))

if __name__ == '__main__':
    if len(sys.argv) == 3:
        sentences = read_lines_from_file(sys.argv[1])
        dictionary = read_lines_from_file(sys.argv[2], keepcase=False)
        test_maxmatch(sentences, dictionary)
    else:
        raise ValueError('Invalid number of arguments: %d' % len(sys.argv))

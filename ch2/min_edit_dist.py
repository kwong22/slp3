#!/usr/bin/python

import numpy as np
import sys

"""
Usage:
    python min_edit_dist.py source target
        Computes minimum edit distance between source and target words.
        The possible edit operations are insertion, deletion, and substitution.
        Each edit operation has a cost of 1.

    source: source word
    target: target word
"""

def to_upper(x):
    """
    Convert input to uppercase.

    Inputs:
    - x: string or list of strings

    Returns:
    - uppercase version of x
    """

    if isinstance(x, str):
        return x.upper()
    elif isinstance(x, list):
        return [el.upper() for el in x]
    else:
        raise TypeError('Input is not type str or list')

def minimum_edit_distance(x, y):
    """
    Compute the minimum edit distance between two sequences (case-insensitive).

    Inputs:
    - x: first sequence (string or list of strings)
    - y: second sequence (string or list of strings)

    Returns:
    - the minimum edit distance between the two sequences
    """
    N, M = len(x), len(y)
    X, Y = to_upper(x), to_upper(y) # case-insensitive

    # Cost values for edit operations
    ins_cost = 1 # insertion cost
    del_cost = 1 # deletion cost
    sub_cost = 1 # substitution cost

    # Create distance matrix, expand to include empty string
    dist_matrix = np.zeros((N+1, M+1), dtype=np.int32)

    # Initialize base cases (zeroth row and zeroth column)
    for i in range(dist_matrix.shape[0]):
        dist_matrix[i, 0] = i
    for j in range(dist_matrix.shape[1]):
        dist_matrix[0, j] = j

    # Calculate distances for rest of matrix
    for i in range(1, dist_matrix.shape[0]):
        for j in range(1, dist_matrix.shape[1]):
            dist_matrix[i, j] = np.amin([dist_matrix[i, j-1] + ins_cost,
                    dist_matrix[i-1, j] + del_cost,
                    dist_matrix[i-1, j-1] + (X[i-1] != Y[j-1]) * sub_cost])

    # Return distance at end of matrix
    return dist_matrix[N, M]

# Only run if executed directly, not when imported
if __name__ == '__main__':
    if len(sys.argv) == 3:
        print(minimum_edit_distance(sys.argv[1], sys.argv[2]))
    else:
        raise ValueError('Invalid number of arguments: %d' % len(sys.argv))

#!/usr/bin/python

import numpy as np
import sys

def minimum_edit_distance(x, y):
    """
    Compute the minimum edit distance between two strings (case-insensitive).

    Inputs:
    - x: first string
    - y: second string

    Returns:
    - the minimum edit distance between the two strings
    """
    N, M = len(x), len(y)
    X, Y = x.upper(), y.upper() # case-insensitive

    # Cost values for edit operations
    ins_cost = 1 # insertion cost
    del_cost = 1 # deletion cost
    sub_cost = 2 # substitution cost

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

if len(sys.argv) == 3:
    print(minimum_edit_distance(sys.argv[1], sys.argv[2]))
else:
    raise ValueError('Invalid number of arguments: %d' % len(sys.argv))

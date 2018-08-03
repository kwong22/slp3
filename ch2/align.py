#!/usr/bin/python

from enum import Enum
import numpy as np
import sys

class Operation(Enum):
    INS = 1 # insertion
    DEL = 2 # deletion
    SUB = 3 # substitution, different letters
    SUB0 = 4 # substitution, same letter

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

def align(x, y):
    """
    Compute the alignment of two strings using minimum edit distance
    (case-insensitive).

    Inputs:
    - x: first string
    - y: second string

    Returns a tuple of:
    - string representation of first string after alignment
    - string representation of second string after alignment
    - string representation of operations taken during alignment
    """
    N, M = len(x), len(y)
    X, Y = x.upper(), y.upper() # case-insensitive

    # Cost values for edit operations
    ins_cost = 1 # insertion cost
    del_cost = 1 # deletion cost
    sub_cost = 1 # substitution cost

    # Create distance matrix, expand to include empty string
    dist_matrix = np.zeros((N+1, M+1), dtype=np.int32)
    # Create matrix to track the operation taken to get to each position
    op_matrix = np.zeros((N+1, M+1), dtype=np.int32)

    # Initialize base cases (zeroth row and zeroth column)
    for i in range(dist_matrix.shape[0]):
        dist_matrix[i, 0] = i
        op_matrix[i, 0] = Operation.DEL
    for j in range(dist_matrix.shape[1]):
        dist_matrix[0, j] = j
        op_matrix[0, j] = Operation.INS

    # Calculate distances for rest of matrix
    for i in range(1, dist_matrix.shape[0]):
        for j in range(1, dist_matrix.shape[1]):
            # Create list of operations and resulting distances
            dists =  []
            dists.append((Operation.INS, dist_matrix[i, j-1] + ins_cost))
            dists.append((Operation.DEL, dist_matrix[i-1, j] + del_cost))

            if X[i-1] == Y[j-1]:
                dists.append((Operation.SUB0, dist_matrix[i-1, j-1]))
            else:
                dists.append((Operation.SUB, dist_matrix[i-1, j-1] + sub_cost))

            # Choose tuple with minimum distance
            min_tup = min(dists, key=lambda tup: tup[1])

            # Update distance and operation matrices
            dist_matrix[i, j] = min_tup[1]
            op_matrix[i, j] = min_tup[0]

    # For visualization of the alignment
    string1 = []
    string2 = []
    alignment = []

    # Backtrace through operations until 0,0 to get the alignment
    i, j = N, M
    while (i > 0) | (j > 0):
        op = op_matrix[i, j]

        if op == int(Operation.INS):
            string1.insert(0, '*')
            string2.insert(0, Y[j-1])
            alignment.insert(0, 'i')
            j -= 1
        elif op == int(Operation.DEL):
            string1.insert(0, X[i-1])
            string2.insert(0, '*')
            alignment.insert(0, 'd')
            i -= 1
        elif op == int(Operation.SUB):
            string1.insert(0, X[i-1])
            string2.insert(0, Y[j-1])
            alignment.insert(0, 's')
            i -= 1
            j -= 1
        elif op == int(Operation.SUB0):
            string1.insert(0, X[i-1])
            string2.insert(0, Y[j-1])
            alignment.insert(0, ' ')
            i -= 1
            j -= 1
        else:
            raise ValueError('Invalid operation: %d' % op)

    return ''.join(string1), ''.join(string2), ''.join(alignment)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        str1, str2, alignment = align(sys.argv[1], sys.argv[2])
        print(str1)
        print('|' * len(str1))
        print(str2)
        print(alignment)
    else:
        raise ValueError('Invalid number of arguments: %d' % len(sys.argv))

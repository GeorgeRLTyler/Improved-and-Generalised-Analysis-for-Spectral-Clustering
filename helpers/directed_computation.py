from typing import List
import numpy as np

def compute_volume(S: list, degrees: np.ndarray):
    return np.sum([degrees[i] for i in S])

def compute_weight_between_sets(S1:list, S2:list, A: np.ndarray):
    return np.sum([A[i,j] for i in S1 for j in S2])

def Psi(partition: List[List[int]], A: np.ndarray, degrees: np.ndarray, C: List[tuple]):
    k = len(partition)
    volume = np.sum(degrees)
    weight_total = 0
    for i in range(k):
        for j in range(k):
             if (i,j) not in C:
                 weight = compute_weight_between_sets(partition[i], partition[j], A)
                 weight_total += weight
    return weight_total/volume


def compute_theta(partition: List[List[int]], A: np.ndarray, degrees: np.ndarray, C: List[tuple]):
    k = len(partition)
    N = len(A)
    weight_total = 0
    for i in range(k):
        for j in range(k):
            if (i, j) in C:
                vol_i = compute_volume(partition[i], degrees)
                vol_j = compute_volume(partition[j], degrees)
                weight = compute_weight_between_sets(partition[i], partition[j], A) / (vol_i + vol_j)
                weight_total += weight
    return weight_total


def compute_new_bound(eigvals, ups):
    b1 = (4 * ups - eigvals[0]) / (eigvals[1] - eigvals[0])
    b2 = (4 * ups) / (eigvals[1])
    return np.min([b1, b2])


def compute_ls_bounds(eigvals, theta, k):
    gamma = eigvals[1] / (1 - (4 / k) * theta)
    bound_1 = 1 / gamma
    bound_2 = 1 / (gamma - 1)
    return bound_1, bound_2
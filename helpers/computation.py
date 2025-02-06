import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from itertools import combinations
from graph import Graph
from helpers.converters import get_laplacian_matrix, get_degree_matrix


def compute_k_way_estimate(normalised_L, indicator_vectors, K):
    k_way_possibilities = []
    assert indicator_vectors.shape[1] == K, 'Indicator vectors should have K columns'
    for i in range(K):
        indicator = indicator_vectors[:, i]
        val = indicator.T @ normalised_L @ indicator
        k_way_possibilities.append(val)
    return max(k_way_possibilities)


def knn_adjacency_matrix(features, k, metric='euclidean'):
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Compute the pairwise distances
    distances = pairwise_distances(features, metric=metric)

    # Get the indices of the k-nearest neighbors for each point
    knn_indices = np.argsort(distances, axis=1)[:, 1:k + 1]

    # Initialize the adjacency matrix
    n_points = features.shape[0]
    adjacency_matrix = np.zeros((n_points, n_points), dtype=float)

    # Fill the adjacency matrix
    for i in range(n_points):
        for j in knn_indices[i]:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Ensure the graph is undirected

    return adjacency_matrix


def get_thresholded_correlation_matrix(features, threshold):
    correlation_matrix = np.corrcoef(features)

    # Create a graph using the correlation matrix
    correlation_matrix[correlation_matrix < threshold] = 0
    return correlation_matrix


def get_normalised_laplacian(A):
    degrees = np.sum(A, axis=1)
    D_inv_sqrt = np.diag([1 / np.sqrt(d) for d in degrees])
    return np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt


def k_means_indicator_vectors(eigenvectors, K):
    assert eigenvectors.shape[1] >= K, 'Number of eigenvectors should be greater than or equal to K'
    kmeans = KMeans(n_clusters=K, random_state=0).fit(eigenvectors)
    indicator_vectors = np.zeros((eigenvectors.shape[0], K))
    for i in range(K):
        indicator_vectors[:, i] = kmeans.labels_ == i
    return indicator_vectors

def get_normalised_projected_indicator_vectors(eigenvectors, indicator_vectors, K):
    beta_K_by_K = indicator_vectors.T @ eigenvectors[:,0:K]
    combined_indicator_vectors = indicator_vectors @ beta_K_by_K
    orthogonalised_vectors = gram_schmidt(combined_indicator_vectors)
    return orthogonalised_vectors

def gram_schmidt(vectors):
    K = vectors.shape[1]
    vecs = vectors.copy()
    for i in range(K):
        vecs[:, i] = vecs[:, i] / np.linalg.norm(vecs[:, i])
        for j in range(i):
            vecs[:, i] = vecs[:, i] - (vecs[:, j].T @ vecs[:, i]) * vecs[:, j]
        vecs[:, i] = vecs[:, i] / np.linalg.norm(vecs[:, i])
    return vecs


def degree_correction(vectors, D_sqrt):
    vectors_corrected = vectors.copy()
    for i in range(vectors.shape[1]):
        vectors_corrected[:, i] = D_sqrt @ vectors[:, i]
        vectors_corrected[:, i] = vectors_corrected[:, i] / np.linalg.norm(vectors_corrected[:, i])
    return vectors_corrected


def compute_rayleigh_quotients(normalised_L, indicator_vectors, K):
    rayleigh_quotients = []
    assert indicator_vectors.shape[1] == K, 'Indicator vectors should have K columns'
    for i in range(K):
        indicator = indicator_vectors[:, i]
        val = (indicator.T @ normalised_L @ indicator) / (indicator.T @ indicator)
        rayleigh_quotients.append(val)
    return rayleigh_quotients


def dfs(node, adj_matrix, visited, component):
    visited.add(node)
    component.append(node)
    for neighbor, is_connected in enumerate(adj_matrix[node]):
        if is_connected and neighbor not in visited:
            dfs(neighbor, adj_matrix, visited, component)


def largest_connected_component(adj_matrix):
    n = len(adj_matrix)
    visited = set()
    components = []

    # Find all connected components
    for node in range(n):
        if node not in visited:
            component = []
            dfs(node, adj_matrix, visited, component)
            components.append(component)

    # Identify the largest connected component
    largest_cc = max(components, key=len)

    # Create the adjacency matrix for the largest connected component
    size = len(largest_cc)
    largest_cc_matrix = np.zeros((size, size), dtype=int)

    node_index = {node: i for i, node in enumerate(largest_cc)}
    for i in largest_cc:
        for j in largest_cc:
            if adj_matrix[i][j]:
                largest_cc_matrix[node_index[i]][node_index[j]] = 1

    return largest_cc_matrix, largest_cc

def apply_recursive_st(rayleigh_quotients, eigenvalues, start, end, error):
    R = rayleigh_quotients[start:end]
    l = end-start
    val = np.sum(R) - l * eigenvalues[start] + eigenvalues[end] * error
    if (eigenvalues[end] != eigenvalues[start]) and (eigenvalues[end] != 0):
        val = val / (eigenvalues[end] - eigenvalues[start])
        return val
    else:
        return 0

def apply_recursive_st_given_indices(rayleigh_quotients, eigenvalues, indices):
    values = []
    for i, i_add_1 in zip(indices, indices[1:]):
        error = np.sum(values)
        val = apply_recursive_st(rayleigh_quotients, eigenvalues, i, i_add_1, error)
        values.append(val)
    return values

def generate_increasing_lists(K):
    """
    Generate all lists of increasing integers from 0 to K.

    Parameters:
    K (int): A positive integer.

    Returns:
    list: A list of lists containing increasing sequences from 0 to K.
    """
    if K < 0:
        raise ValueError("K must be a positive integer")

    # Create the range of numbers from 0 to K
    numbers = list(range(K + 1))

    # Generate all combinations of indices (at least 2 elements: 0 and K are mandatory)
    result = []
    for r in range(2, len(numbers) + 1):
        for combination in combinations(numbers[1:-1], r - 2):
            result.append([0] + list(combination) + [K])

    return result
#%%
def apply_recursive_st_brute_force(rayleigh_quotients, eigenvalues, K):
    list_of_indice_splits = generate_increasing_lists(K)
    min_split = np.inf
    min_split_indices = list_of_indice_splits[0]
    for indices_split in list_of_indice_splits:
        val = np.sum(apply_recursive_st_given_indices(rayleigh_quotients, eigenvalues, indices_split))
        if val < min_split:
            min_split = val
            min_split_indices = indices_split
    return min_split_indices, min_split


import logging
from typing import List, Union
import numpy as np

from graph import Graph

logger = logging.getLogger(__name__)


def get_adjacency_matrix(G: Graph) -> np.array:
    N = len(G.vertices)
    edges = G.edges
    adjacency_matrix = np.zeros((N, N))
    if G.directed:
        if G.weighted:
            weights = G.weights
            for edge in edges:
                adjacency_matrix[edge] = weights[edge]
        else:
            for edge in edges:
                adjacency_matrix[edge] = 1
    else:
        if G.weighted:
            weights = G.weights
            for edge in edges:
                adjacency_matrix[edge] = weights[edge]
                adjacency_matrix[(edge[::-1])] = weights[edge]
        else:
            for edge in edges:
                adjacency_matrix[edge] = 1
                adjacency_matrix[(edge[::-1])] = 1
    return adjacency_matrix


def get_hermitian_adjacency_matrix(G: Graph, root_of_unity: int = 4) -> np.array:
    if not G.directed:
        logger.warning("Graph is not directed. Computing hermitian adjacency matrix of undirected graph.")
        return get_adjacency_matrix(G)

    N = len(G.vertices)
    adjacency_matrix = np.zeros((N, N), dtype=np.complex128)
    w_k = np.exp(2 * np.pi * 1j / root_of_unity)
    w_k_conj = np.conj(w_k)
    if G.weighted:
        weights = G.weights
        for edge in G.edges:
            adjacency_matrix[edge] = weights[edge] * w_k
            adjacency_matrix[(edge[::-1])] = weights[edge] * w_k_conj
    else:
        for edge in G.edges:
            adjacency_matrix[edge] = w_k
            adjacency_matrix[(edge[::-1])] = w_k_conj
    return adjacency_matrix


def get_degree_matrix(G: Graph) -> Union[np.array, List[np.array]]:
    N = len(G.vertices)
    edges = G.edges
    if G.directed:
        in_degree = np.zeros(N)
        out_degree = np.zeros(N)
        if G.weighted:
            weights = G.weights
            for edge in edges:
                out_degree[edge[0]] += weights[edge]
                in_degree[edge[1]] += weights[edge]
        else:
            for edge in edges:
                out_degree[edge[0]] += 1
                in_degree[edge[1]] += 1
        return np.diag(out_degree), np.diag(in_degree)
    else:
        degree = np.zeros(N)
        if G.weighted:
            weights = G.weights
            for edge in edges:
                degree[edge[0]] += weights[edge]
                degree[edge[1]] += weights[edge]
        else:
            for edge in edges:
                degree[edge[0]] += 1
                degree[edge[1]] += 1
        return np.diag(degree)


def get_laplacian_matrix(G: Graph, normalized=False) -> np.array:

    if G.directed:
        logger.warning("Graph is directed. Computing laplacian matrix of directed graph.")
    adjacency_matrix = get_adjacency_matrix(G)
    degree_matrix = get_degree_matrix(G)
    laplacian = degree_matrix - adjacency_matrix
    if normalized:
        degree_matrix = np.diag(1 / np.sqrt(np.diag(degree_matrix)))
        laplacian = degree_matrix @ laplacian @ degree_matrix
    return laplacian


def get_signless_laplacian_matrix(G: Graph, normalized=False) -> np.array:
    if G.directed:
        logger.warning("Graph is directed. Computing signless laplacian matrix of directed graph.")
    adjacency_matrix = get_adjacency_matrix(G)
    degree_matrix = get_degree_matrix(G)
    signless_laplacian = degree_matrix + adjacency_matrix
    if normalized:
        degree_matrix = np.diag(1 / np.sqrt(np.diag(degree_matrix)))
        signless_laplacian = degree_matrix @ signless_laplacian @ degree_matrix
    return signless_laplacian
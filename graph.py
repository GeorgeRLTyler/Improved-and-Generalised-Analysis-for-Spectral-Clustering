import numpy as np

class Graph:
    def __init__(self, vertices=None, edges=None, weighted=False, directed=False):
        self.vertices = vertices
        self.edges = edges
        self.weighted = weighted
        self.directed = directed

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def add_edge(self, edge, weight=None):
        if self.weighted:
            self.edges.append(edge)
            self.weights[edge] = weight
        else:
            self.edges.append(edge)
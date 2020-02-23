"""an
Script with various graph operations.
"""

import math
from collections import defaultdict
import copy


def topological_sort(graph):
    """
    Returns a list that contains the vertices of graph, topologically ordered.\
    A DAG is assumed as input.
    """

    def topological_sort_util(vertex):
        """Helper function used to implement topological sorting"""

        # Mark current vertex as visited
        visited[vertex] = True

        # Iterate all adjacent vertices
        for v in graph[vertex]:
            if not visited[v]:
                topological_sort_util(v)

        top_sort_list.insert(0, vertex)

    # If self.graph or self.cpts not initialized
    if not graph:
        print("Graph empty")
        return None

    visited = {}  # dictionary that keeps which vertices have been met
    top_sort_list = []  # list that contains vertices topologically sorted

    # Initialize all vertices as unvisited
    for key in graph:
        visited[key] = False

    # Iterate all vertices
    for u in graph:
        if not visited[u]:
            topological_sort_util(u)

    return top_sort_list


def prim(graph, vertices, min_max):
    """
    Prim's algorithm to find minimum spanning tree of graph.
    :param min_max: indicates whether to construct a minimum or maximal spanning tree
    :param vertices: the names of vertices (ordered)
    :param graph: adjacency matrix as a 2d numpy array
    :return min_st: the maximum spanning tree as a dictionary (representation of an undirected graph)
    """

    def min_key():

        min_val = math.inf  # initialize maximum weight
        min_idx = -1          # initialize return index

        # for all the vertices
        for i in range(len(vertices)):
            if not mst_set[i] and keys[i] < min_val:
                min_val = keys[i]
                min_idx = i

        return min_idx

    graph_c = copy.deepcopy(graph)
    if min_max == 'max':
        graph_c = -graph_c     # negative weights to construct maximal spanning tree

    min_st = [0] * len(vertices)        # mst vertices
    keys = [math.inf] * len(vertices)   # prim's keys
    mst_set = [False] * len(vertices)   # list that indicates which vertices have been added to the tree

    # initialize first vertex's key
    keys[0] = 0

    # while there are vertices to include
    while mst_set != [True] * len(vertices):
        min_v_idx = min_key()
        mst_set[min_v_idx] = True

        # iterate all adjacent vertices of vertex with maximum key
        for u_idx in range(len(graph_c[min_v_idx, :])):

            if graph_c[min_v_idx, u_idx] and keys[u_idx] > graph_c[min_v_idx, u_idx] and not mst_set[u_idx]:
                keys[u_idx] = graph_c[min_v_idx, u_idx]
                min_st[u_idx] = vertices[min_v_idx]

    # return mst as a dictionary
    min_st_dict = defaultdict(list)
    for j in range(1, len(vertices)):
        min_st_dict[min_st[j]].append(vertices[j])
        min_st_dict[vertices[j]].append(min_st[j])

    min_st_dict = dict(min_st_dict)

    return min_st_dict

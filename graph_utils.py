"""
Script with various graph operations. A graph is assumed to be a dictionary with vertices as keys and adjacency
lists as values
"""


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

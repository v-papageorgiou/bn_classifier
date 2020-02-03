import json
import numpy as np


class BayesianNetwork:
    """
    ========================== DOC ==================
    """

    def __init__(self, file_name):
        self.graph = {}     # the graph that corresponds to the bayesian network
        self.cpts = {}      # a dict that holds the conditional probability table of each vertex

        self.generate_network(file_name)

    def generate_network(self, file_name):
        """"
        Generates the graph alongside the conditional probability tables that correspond to the file of the user.
        Correct json format as well as cpt and graph relations are assumed

        :param file_name: the name of the input file
        :return: a tuple with two dictionaries:
            graph: a dictionary with all the vertices of the graph as keys and their adjacency lists as values
            cpts: a dictionary with all the vertices as keys and the corresponding cpt values in the form of numpy arrays as
                values
        """

        self.graph = {}
        self.cpts = {}
        try:
            # Read the data from the json file
            with open(file_name, 'r') as file:
                data = json.load(file)

            # Fill dictionaries graph and cpts
            for key in data:
                self.graph[key] = data[key]['adj']  # adjacency list
                if not data[key]['parents']:        # case of no parent vertices
                    cpt = ['prob']
                    cpt.extend(data[key]['cpt'])
                    cpt = np.array(cpt)

                else:
                    cpt = [data[key]['parents']]        # list of parent vertices
                    cpt.extend(data[key]['cpt'])        # merge parent vertices and cpt values
                    cpt[0].append('prob')               # append the name of the last column ('prob')
                cpt = np.array(cpt)
                self.cpts[key] = cpt

        except IOError:
            print('Could not open file: {}'.format(file_name))

    def topological_sort(self):
        """Returns a list that contains the vertices of self.graph, topologically ordered"""

        def topological_sort_util(self, vertex):
            """Helper function used to implement topological sorting"""

            # Mark current vertex as visited
            visited[vertex] = True

            # Iterate all adjacent vertices
            for v in self.graph[vertex]:
                if not visited[v]:
                    topological_sort_util(self, v)

            top_sort_list.insert(0, vertex)

        # If self.graph or self.cpts not initialized
        if not self.graph or not self.cpts:
            print("Graph or CPTs empty")
            return None

        visited = {}        # dictionary that keeps which vertices have been met
        top_sort_list = []  # list that contains vertices topologically sorted

        # Initialize all vertices as unvisited
        for key in self.graph:
            visited[key] = False

        # Iterate all vertices
        for u in self.graph:
            if not visited[u]:
                topological_sort_util(self, u)

        return top_sort_list

    def generate_dataset(self, n_samples, file_name):
        """
        Generates a synthetic dataset in csv format, using prior sampling to the current Bayesian Network

        :param
            n_samples: The number of samples that are to be created
            file_name: The name of the file that the data are to be written to
        """

        top_sort_v = self.topological_sort()    # vertices topologically sorted
        try:
            with open(file_name + '.csv', 'w') as file:
                file.write(','.join(top_sort_v) + '\n')
                for _ in range(n_samples):
                    sample = self.prior_sampling(top_sort_v)
                    values = [str(val) for val in list(sample.values())]
                    file.write(','.join(values) + '\n')

        except IOError:
            print('Error trying to perform file IO')

    def prior_sampling(self, top_sort_v):
        """
        :param:
            top_sort_v: A list with all the vertices of the graph, topologically ordered
        :return: A sample of the joint distribution of all the random variables of the network
        """

        # Dictionary that holds the value of each
        # random for the current sample
        sample = {}
        for u in top_sort_v:    # initialization
            sample[u] = False

        # Iterate all vertices
        for u in sample:

            # case of vertex with no parents
            if len(self.cpts[u]) == 2:
                sample[u] = np.random.rand() <= float(self.cpts[u][1])

            # case of vertex with parents
            else:
                parents = self.cpts[u][0, 0:-1]     # parent vertices

                parent_values = []  # boolean list that holds the values of each parent for the current sample
                for v in parents:
                    parent_values.append(sample[v])

                # Calculation of row index in cpt
                bin_val = 1
                idx = 1
                for p_val in parent_values[::-1]:
                    idx += (1-p_val)*bin_val
                    bin_val <<= 1

                prob = float(self.cpts[u][idx, len(parents)])  # probability of the current random variable to be True
                sample[u] = np.random.rand() <= prob

        return sample


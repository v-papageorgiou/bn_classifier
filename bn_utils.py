
import graph_utils
import json
import random
import copy
import numpy as np


def normalize(q):
    """
    Normalizes the values of q, so that they add up to 1
    :param q: A distribution
    :return: The values of q, so that they are probability values
    """
    return [round(norm_q / sum(q), 8) for norm_q in q]


class BayesianNetwork:
    """
    ========================== DOC ==================
    """

    def __init__(self, file_name='', graph=None, cpts=None):
        self.graph = {}     # the graph that corresponds to the bayesian network
        self.cpts = {}      # a dict that holds the conditional probability table of each vertex

        # if graph and cpts are given
        if graph and cpts:
            self.graph = graph
            self.cpts = cpts
        else:
            self.generate_network(file_name)

    def generate_network(self, file_name):
        """
        Generates the graph alongside the conditional probability tables that correspond to the file of the user.
        Correct json format as well as cpt and graph relations are assumed

        :param file_name: the name of the input file
        :return: a tuple with two dictionaries:
            graph: a dictionary with all the vertices of the graph as keys and their adjacency lists as values
            cpts: a dictionary with all the vertices as keys and the corresponding cpt values as values
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

                else:
                    cpt = [data[key]['parents']]        # list of parent vertices
                    cpt.extend(data[key]['cpt'])        # merge parent vertices and cpt values
                    cpt[0].append('prob')               # append the name of the last column ('prob')

                self.cpts[key] = cpt

        except IOError:
            print('Could not open file: {}'.format(file_name))

    def generate_dataset(self, n_samples, file_name):
        """
        Generates a synthetic dataset in csv format, using prior sampling to the current Bayesian Network

        :param
            n_samples: The number of samples that are to be created
            file_name: The name of the file that the data are to be written to
        """

        top_sort_v = graph_utils.topological_sort(self.graph)    # vertices topologically sorted
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
                sample[u] = random.random() <= float(self.cpts[u][1])

            # case of vertex with parents
            else:
                parents = self.cpts[u][0][0:-1]     # parent vertices

                parent_values = []  # boolean list that holds the values of each parent for the current sample
                for v in parents:
                    parent_values.append(sample[v])

                # Calculation of row index in cpt
                bin_val = 1
                idx = 1
                for p_val in parent_values[::-1]:
                    idx += (1-p_val)*bin_val
                    bin_val <<= 1

                prob = float(self.cpts[u][idx][len(parents)])  # probability of the current random variable to be True
                sample[u] = random.random() <= prob

        return sample

    def enumeration_ask(self, x, evidence):
        """
        Enumeration algorithm for exact inference in a Bayesian Network, given the appropriate evidence

        :param x: The random variable to which the query refers
        :param evidence: A dictionary with vertices as keys and their boolean values as values
            (i.e {'varA':True, 'varB':False})
        :return: A distribution over x, named q
        """

        # if x or evidence in None or x is not a network variable
        if not x or not evidence or x not in self.graph:
            print('Invalid arguments')
            return None

        # if some evidence variable is not part of the network
        for e in evidence:
            if e not in self.graph:
                print('Invalid arguments')
                return None

        q = []                                                  # the distribution of x
        e = copy.deepcopy(evidence)                             # deep copy the evidence
        variables = graph_utils.topological_sort(self.graph)    # bayesian network variables topologically ordered

        # for all the possible values of x
        for val in [True, False]:

            # extend evidence with current x value
            e[x] = val

            q.append(self.enumerate_all(variables, e))

        return normalize(q)

    def enumerate_all(self, variables, evidence):
        """
        Enumerate over all variables.

        :param evidence:
        :param variables:
        :return:
        """
        if not variables:
            return 1.0

        y = variables[0]    # get first variable
        if y in evidence:   # if y has a value in evidence
            return self.probability(y, evidence) * self.enumerate_all(variables[1:], evidence)

        else:               # if y has no value in evidence
            ey = copy.deepcopy(evidence)     # deepcopy evidence
            probabilities = []
            for val in [True, False]:
                ey[y] = val  # add y evidence to e
                probabilities.append(self.probability(y, ey) * self.enumerate_all(variables[1:], ey))

            return sum(probabilities)

    def probability(self, y, evidence):
        """
        ============================= DOC =============================================================================================================================================
        :param y:
        :param evidence:
        :return:
        """
        if y not in self.graph:
            print('{} is not in graph'.format(y))
            return None

        # if y has no parents
        if len(self.cpts[y]) == 2:
            return self.cpts[y][1] if evidence[y] else 1 - self.cpts[y][1]

        # if y has parents
        else:
            cpt_y = np.array(self.cpts[y])      # cpt of y as numpy array
            parents = list(cpt_y[0, :-1])       # parents of y
            parent_evidence = []
            for p in parents:
                parent_evidence.append(bool(evidence[p]))

            parent_evidence = np.array(parent_evidence)

            # boolean numpy array that holds to which evidence does parent evidence refer
            row = np.array(2**len(parent_evidence)*[True])
            for i in range(len(parent_evidence)):
                row = np.logical_and(cpt_y[1:, i] == str(parent_evidence[i]), row)

            row_idx = int(np.where(row)[0]) + 1
            return self.cpts[y][row_idx][-1] if evidence[y] else 1 - self.cpts[y][row_idx][-1]
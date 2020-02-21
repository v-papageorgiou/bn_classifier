from ctypes import py_object

import pandas as pd
import numpy as np
import copy
import math
import graph_utils
import random
import abc


class BNClassifier(abc.ABC):

    @abc.abstractmethod
    def __init__(self, file_name, label, train_percentage=1.0):

        # read the csv file and create a pandas data frame
        try:
            self.data_frame = pd.read_csv(file_name)
            self.data_frame = self.data_frame.head(self.data_frame.shape[0] * math.floor(train_percentage))
        except IOError:
            print('Error reading file: {}'.format(file_name))
            return

        # label of classification
        self.label = label

        # names of features
        self.features = list(self.data_frame.columns)
        self.features.remove(label)

        # learnt structure and cpts
        self.graph = {}
        self.cpts = {}

    # @abc.abstractmethod
    # def maximum_likelihood(self):
    #     """
    #     Calculates maximum likelihood of feature v
    #     """
    #     pass


class TANClassifier(BNClassifier):
    """
    Tree Augmented Naive Bayes classifier
    """

    def __init__(self, file_name, label, train_percentage=1.0):
        super().__init__(file_name, label, train_percentage)

        _, cols = self.data_frame.shape  # number of features (label included)
        self.adjacency_matrix = np.zeros((cols - 1, cols - 1))  # initialization of adjacency matrix (label excluded)
        self.generate_adjacency_matrix()
        self.build_classifier()

    def generate_adjacency_matrix(self):
        """
        Fills self.adjacency_matrix with mutual information of different features given the label as weights.
        """

        # iterate all pairs of features, calculate the corresponding conditional mutual information and
        # assign it as the weight between i-th and j-th feature (and vice versa since the graph is undirected)
        for i in range(len(self.features)):
            for j in range(i + 1, len(self.features)):
                mut_info = self.conditional_mutual_information(self.features[i], self.features[j],
                                                               self.data_frame.shape[0])
                self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i] = mut_info

    def conditional_mutual_information(self, feature_1, feature_2, n_samples):
        """
        Calculates the conditional mutual information of random variables features_1 and features_2 given the random
        variable self.label and the number of total samples of the dataset, n_sample
        :param feature_1: first random variable
        :param feature_2: second random variable
        :param n_samples: number of samples
        :return: conditional mutual information
        """
        cols = copy.deepcopy(self.data_frame)[[feature_1, feature_2, self.label]]  # get features' and label's columns
        mutual_info = 0  # return value
        for val_label in [True, False]:
            for val2 in [True, False]:
                for val1 in [True, False]:
                    # probability of joint feature_1, feature_2, label random variables
                    p_12l = 10 ** -8 + cols[
                        (cols[feature_1] == val1) & (cols[feature_2] == val2) & cols[self.label] == val_label].shape[
                        0] / n_samples

                    # probability of label
                    p_l = 10 ** -8 + cols[(cols[self.label] == val_label)].shape[0] / n_samples

                    # probability of joint feature_1, label
                    p_1l = 10 ** -8 + cols[(cols[feature_1] == val1) & (cols[self.label] == val_label)].shape[
                        0] / n_samples

                    # probability of join feature_2, label
                    p_2l = 10 ** -8 + cols[(cols[feature_2] == val2) & (cols[self.label] == val_label)].shape[
                        0] / n_samples

                    mutual_info += p_12l * math.log2((p_12l * p_l) / (p_1l * p_2l))

        return mutual_info

    def chow_liu(self):
        """
        Implementation of chow_liu algorithm that returns a directed maximum spanning tree, using the conditional mutual
        information as edge weight
        :return max_st: the maximal spanning tree
        """

        def give_directions(vertex):
            """
            Gives directions to max_st using DFS
            """
            # check if all vertices are marked
            # if sum([_v for _v in marked.values()]) == len(marked):
            #     return

            for u in max_st[vertex]:
                if not marked[u]:  # if adjacent vertex is not marked
                    marked[u] = True  # mark it
                    if max_st[u]:  # delete edge between u and vertex
                        max_st[u].remove(vertex)
                    give_directions(u)

        max_st = graph_utils.prim(self.adjacency_matrix, self.features, 'max')  # maximal spanning tree
        marked = {}  # dictionary that holds which vertices have been met during give_directions() (DFS)
        for v in max_st:
            marked[v] = False

        init_v = random.choice(list(marked))
        marked[init_v] = True
        give_directions(init_v)
        return max_st

    def maximum_likelihood(self, v, subgraph, marked, parent):
        """
        Calculates the cpts (self.cpts) of self. graph using the maximum liklihood rule
        :param v: The vertex whose cpt is being calculated
        :param subgraph: The subtree of self.graph that only contains the features
        :param marked: A dict which holds which vertices have been processed
        :param parent: Parent of v
        """

        if marked[v]:  # if current vertex's cpt has been calculate
            return

        # calculate maximum likelihood of all adjacent vertices recursively
        for u in subgraph[v]:
            self.maximum_likelihood(u, subgraph, marked, v)

        marked[v] = True  # mark current vertex
        self.calc_cpt(v, parent)  # calculate cpt

    def calc_cpt(self, v, parent):
        """
        Calculates cpt of v given its parent. Note that the topology of a TAN classifier bayesian network suggests that
        each vertex has at most two parents, including the vertex of the target label.
        :param v: The vertex whose cpt is being calculated
        :param parent: The (only) parent of v
        """
        self.cpts[v] = []

        if not parent:  # if the only parent of the vertex is the label

            # if we calculate the cpt of the label vertex
            if v == self.label:
                data = copy.deepcopy(self.data_frame[v])
                p_true = data.sum() / data.shape[0]
                self.cpts[v].append('prob')
                self.cpts[v].append(p_true)
                return

            # get columns of label and v
            data = copy.deepcopy(self.data_frame[[self.label, v]])
            self.cpts[v].append([self.label, 'prob'])

            # for all the values of the label
            for val_label in [True, False]:
                # get data that correspond to val_label
                data_label = data[data[self.label] == val_label]

                # probability of v given the value of the label
                p_true = data_label[data_label[v] == True].shape[0] / data_label.shape[0]
                self.cpts[v].append([val_label, p_true])
        else:

            # get columns of label, parent and v
            data = copy.deepcopy(self.data_frame[[self.label, parent, v]])
            self.cpts[v].append([self.label, parent, 'prob'])

            for val_label in [True, False]:     # for all the values of the label
                for val_par in [True, False]:   # for the values of parent

                    # get the data that correspond to val_label and val_parent
                    data_label_par = data[(data[self.label] == val_label) & (data[parent] == val_par)]

                    # probability of v give the values of the label and the (only) parent
                    p_true = data_label_par[data_label_par[v] == True].shape[0] / data_label_par.shape[0]
                    self.cpts[v].append([val_label, val_par, p_true])

    def build_classifier(self):

        # learn structure of TAN classifier
        self.graph = self.chow_liu()
        self.graph[self.label] = [v for v in self.graph]

        # learn parameters of TAN classifier
        graph_features = copy.deepcopy(self.graph)  # get feature subgraph
        del graph_features[self.label]  #

        features_ts = graph_utils.topological_sort(graph_features)

        # marked vertices during maximum likelihood
        marked = {}
        for v in graph_features:
            marked[v] = False

        for v in features_ts:
            self.maximum_likelihood(v, graph_features, marked, None)

        # calculate cpt of label
        self.calc_cpt(self.label, None)

        print(self.cpts)

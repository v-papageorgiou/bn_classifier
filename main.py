import bn_utils
import bn_classifier
import pandas as pd
import math
from bn_test import count_misclassified_items
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

model_path = 'models/'
data_path = 'datasets/'


def init():
    print('\n============================================ Bayesian Network Classifiers ============================================')
    print('Networks Loaded:')
    models_ = [f[0:-5] for f in listdir(model_path) if isfile((join(model_path, f)))]
    for i, m in enumerate(models_):
        print(i+1, m)
    print()
    return models_


def plot_data(errors_bn, errors_tan_bn, errors_nb_bn, model):

    dim = len(errors_bn)
    w = 0.5
    dimw = w / dim

    bars = list(errors_bn.keys())
    values = list(errors_bn.values())
    y = np.arange(len(bars))
    plt.bar(y, values, dimw)
    plt.xticks(y, list(errors_bn.keys()))

    bars = list(errors_tan_bn.keys())
    values = list(errors_tan_bn.values())
    y = np.arange(len(bars))
    plt.bar(y + dimw, values, dimw)
    plt.xticks(y, list(errors_tan_bn.keys()))

    bars = list(errors_nb_bn.keys())
    values = list(errors_nb_bn.values())
    y = np.arange(len(bars))
    plt.bar(y + 2*dimw, values, dimw)
    plt.xticks(y, list(errors_nb_bn.keys()))

    plt.title(model)

    plt.legend(['BN error', 'TAN error', 'NB error'])


def choose_label(features):
    print('Choose a label from: ')

    for f in features:
        print('(' + f + ')', end=' ')

    print()
    l = None
    while l not in features:
        l = input('Enter feature: ')

    return l


models = init()
train_percentage = 0.8
n_samples = 10000
iterations = 5

try:
    for model in models:

        print('\nModel being processed: ' + model)
        fig = plt.figure()

        # create the bayesian network that corresponds to input file and generate an artificial data set
        bn = bn_utils.BayesianNetwork(model_path + model + '.json')
        bn.generate_dataset(n_samples, data_path + model)

        # read artificial
        data_frame = pd.read_csv(data_path + model + '.csv')

        # dict that holds the error percentage for each label of each model
        errors_bn = {}
        errors_tan_bn = {}
        errors_nb_bn = {}

        # labels to iterate
        labels = list(data_frame.columns.values)

        if model == 'medicine':
            labels = ['lung_cancer', 'bronchitis', 'pneumonia']
        elif model == 'suicide':
            labels = ['suicide', 'insomnia']

        for label in tqdm(labels, desc='Running experiments for ' + model + ' network'):
            bn_errors = 0
            tan_classifier_errors = 0
            nb_bn_errors = 0
            for i in range(iterations):

                # shuffle the dataset and create train and test data frames
                data_frame = data_frame.sample(frac=1)
                train_data_frame = data_frame.head(math.floor(data_frame.shape[0] * train_percentage))
                test_data_frame = data_frame.tail(math.ceil(data_frame.shape[0] * (1 - train_percentage)))

                # create the classifiers
                tan_classifier = bn_classifier.TANClassifier(train_data_frame, label, False)
                tan_classifier_bn = bn_utils.BayesianNetwork(graph=tan_classifier.graph, cpts=tan_classifier.cpts)
                nb_classifier = bn_classifier.TANClassifier(train_data_frame, label, True)
                nb_classifier_bn = bn_utils.BayesianNetwork(graph=nb_classifier.graph, cpts=nb_classifier.cpts)

                # count misclassified samples from the test dataset
                bn_errors += count_misclassified_items(bn, test_data_frame, label)
                tan_classifier_errors += count_misclassified_items(tan_classifier_bn, test_data_frame, label)
                nb_bn_errors += count_misclassified_items(nb_classifier_bn, test_data_frame, label)

            errors_bn[label] = bn_errors / iterations / test_data_frame.shape[0]
            errors_tan_bn[label] = tan_classifier_errors / iterations / test_data_frame.shape[0]
            errors_nb_bn[label] = nb_bn_errors / iterations / test_data_frame.shape[0]

        plot_data(errors_bn, errors_tan_bn, errors_nb_bn, model)

    plt.show()
except IOError:
    print('IOError raised in main()')
    pass



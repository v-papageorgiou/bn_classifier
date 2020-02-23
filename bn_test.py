

def count_misclassified_items(network, test_data_frame, label):
    """
    Counts the samples where the prediction of the bayesian network with graph as structure and cpts as conditional
    probability tables does not meet with the actual value of the label
    :param network: The bayesian network in the form of an instance of BayesianNetwork class
    :param label: The label that the classification has been built upon
    :param graph: The graph of the network in the form of a dictionary
    :param cpts: The conditional probability tables
    :param test_data_frame: The test data as a pandas dataframe
    :return: the number of misclassified items
    """

    # get the feature names from the dataset
    features = list(test_data_frame.columns.values)
    features.remove(label)
    errors = 0
    for i, row in test_data_frame.iterrows():
        feature_values = list(row)[0:-1]
        evidence = {feature: value for feature, value in zip(features, feature_values)}
        errors += network.enumeration_ask(label, evidence)[0] > 0.5 != list(row)[-1]

    return errors

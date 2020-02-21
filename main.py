import bn_utils
import bn_classifier

bn = bn_utils.BayesianNetwork('suicide.json')
bn.generate_dataset(10000, 'suicide')

classifier = bn_classifier.TANClassifier('suicide.csv', 'su')
classifier = bn_utils.BayesianNetwork(graph=classifier.graph, cpts=classifier.cpts)


print(bn.enumeration_ask('su', {'bu': False, 'nff': False}))
print(classifier.enumeration_ask('su', {'bu': False, 'nff': False}))

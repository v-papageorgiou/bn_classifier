import bn_utils

bn = bn_utils.BayesianNetwork('suicide.json')
bn.generate_dataset(100, 'suicide')

# print(bn.graph)
print(bn.enumeration_ask('su', {'bu': False, 'rl': False, 'dep': False, 'ps': False}))


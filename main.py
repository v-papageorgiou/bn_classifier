import bn_utils
import pandas as pd

# bn = bn_utils.BayesianNetwork('suicide.json')
# bn.generate_dataset(100, 'suicide')


data = pd.read_csv('suicide.csv')
print(data)
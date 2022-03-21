import numpy as np
from sklearn.datasets import load_svmlight_file

path = './data/BIG15_basic_lbph.txt'

data = load_svmlight_file(path)

print(data)
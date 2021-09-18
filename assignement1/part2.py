import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sc

#Load data
data = pd.read_csv('iris.data', names=['sepal length','sepal width', 'petal length', 'petal width'])

#split test and train
train=data.sample(frac=0.75,random_state=200) #random state is a seed value
test=data.drop(train.index)

print(data)
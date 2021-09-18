import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sc

#Load data
data = pd.read_csv('truckdata.txt', names=['x','y'])

#Split test data
train=data.sample(frac=0.8,random_state=200) #random state is a seed value
test=data.drop(train.index)

#print(data)
#print(train)
print(test)
ax = train.plot.scatter(x='x',y='y', color="DarkBlue");
test.plot.scatter(x='x',y='y', color="DarkRed", ax=ax);
plt.show();



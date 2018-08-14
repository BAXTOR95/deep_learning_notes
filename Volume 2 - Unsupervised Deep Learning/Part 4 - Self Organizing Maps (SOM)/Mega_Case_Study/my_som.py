# Self Organizing Map

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import bone, colorbar, pcolor, plot, show
from sklearn.preprocessing import MinMaxScaler

from minisom import MiniSom

# Importing the dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=1000)

# Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']  # 'o' - Circle, 's' - Square
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(6, 4)], mappings[(5, 5)]), axis=0)
frauds = sc.inverse_transform(frauds)

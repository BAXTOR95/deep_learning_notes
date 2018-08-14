# Mega Case Study - Make a Hybrid Deep Learning Model


def eval_performance(cm):
    """eval_performance (cm)

    Function that calculates the performance of a ML Model
    by getting the Accuracy, Precision, Recall and F1 Score values

    Arguments:
        cm {List} -- The Confusion Matrix
    """
    tp = cm[0, 0]
    tn = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"True Positives: {tp}\nTrue Negatives: {tn}\n" +
          f"False Positives: {fp}\nFalse Negatives: {fn}\n")
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\n" +
          f"F1 Score: {f1_score}")



# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Importing the Keras libraries and packages
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json
from pylab import bone, colorbar, pcolor, plot, show
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from minisom import MiniSom

# Part 1 - Identify the Frauds with the Self-Organizing Map

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
frauds = np.concatenate(
    (mappings[(1, 1)], mappings[(6, 5)]), axis=0)
frauds = sc.inverse_transform(frauds)


# Part 2 - Going from Unsupervised to Supervised Deep Learning
#new_dataset = []
#new_dataset = np.concatenate((frauds, np.zeros((len(frauds), 1))), axis=1)
# for i in range(len(new_dataset)-1):
#    for j, x2 in enumerate(sc.inverse_transform(X)):
#        if new_dataset[i, 0] == x2[0]:
#            new_dataset[i, -1] = y[j]
#            break
#        else:
#            continue
#new_X = new_dataset[:, 1:-1]
#new_y = new_dataset[:, -1]

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros((len(customers), 1))
for i in range(len(customers)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

new_sc = MinMaxScaler(feature_range=(0, 1))
customers = new_sc.fit_transform(customers)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer (with dropout)
classifier.add(Dense(activation="relu", input_dim=15,
                     units=2, kernel_initializer="uniform"))
#classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1,
                     kernel_initializer="uniform"))

# Compiling the ANN
# adam correspond to a stochastic gradient descent algorithm
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the probabilities of fraud
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]

y_pred2 = classifier.predict(customers)
y_pred2 = (y_pred2 > 0.2)

# Making the Confusion Matrix
cm = confusion_matrix(is_fraud, y_pred2)

print("\nEvaluating performance for ANN model")
eval_performance(cm)

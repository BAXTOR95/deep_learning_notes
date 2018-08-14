
# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::',
                     header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::',
                    header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::',
                      header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns


def convert(data):
    """convert (data)

    Function that converts an array to a ``list of list``

    Arguments:
        data {Array} -- The data set

    Returns:
        List -- The new data set
    """
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network


class RBM():
    def __init__(self, nv, nh):
        self._w = torch.randn(nh, nv)  # Weights
        self._a = torch.randn(1, nh)  # Bias for Prob h given v (Batch, Bias)
        self._b = torch.randn(1, nv)  # Bias for Prob v given h (Batch, Bias)

    def sample_h(self, x):
        """sample_h (x)

        Function that samples the hidden nodes according to the probability
        of ``p_h_given_v``

        Arguments:
            x {torch.FloatTensor} -- The Visible Nodes

        Returns:
            torch.sigmoid, torch.bernoulli -- Returns the probability
        """
        wx = torch.mm(x, self._w.t())  # wx -> minibatch | t() transpose
        activation = wx + self._a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """sample_v (y)

        Function that samples the visible nodes according to the probability
        of ``p_v_given_h``

        Arguments:
            y {torch.FloatTensor} -- The Hidden Nodes

        Returns:
            torch.sigmoid, torch.bernoulli -- Returns the probability and a
            sample of the hidden nodes
        """
        wy = torch.mm(y, self._w)  # wy -> minibatch
        activation = wy + self._b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        """train (v0, vk, ph0, phk)

        Function that does the constracting divertion with Gibs Sampling

        Arguments:
            v0 {tensor vector} -- Input Vector Containing the ratings of all
                                  the movies by one user (the Observations)
            vk {tensor vector} -- Visible nodes obtained after k-samplings
                                  (k iterations in k contrasting divergence)
            ph0 {tensor vector} -- The vertor of probabilities that at the
                                   first iteration the hidden nodes equal
                                   one given the values of v0
            phk {tensor vector} -- The probabilities of the hidden nodes after
                                   k-samplings given the values of
                                   the visible nodes vk
        Returns:
            None
        """
        self._w += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self._b += torch.sum((v0 - vk), 0)
        self._a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0])
nh = 100
batch_size = 100

rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    rmse_train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        rmse_train_loss += np.sqrt(torch.mean((v0[v0 >= 0] - vk[v0 >= 0])**2))
        s += 1.
    print(f"epoch: {epoch} | Average Distance loss: {train_loss/s:.2f} \
          | RMSE: {rmse_train_loss/s:.2f}")

# Testing the RBM
test_loss = 0
rmse_test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]
    vt = test_set[id_user:id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        rmse_test_loss += np.sqrt(torch.mean((vt[vt >= 0] - v[vt >= 0])**2))
        s += 1.
print(f"Average Distance loss: {test_loss/s:.2f} \
      | RMSE: {rmse_test_loss/s:.2f}")

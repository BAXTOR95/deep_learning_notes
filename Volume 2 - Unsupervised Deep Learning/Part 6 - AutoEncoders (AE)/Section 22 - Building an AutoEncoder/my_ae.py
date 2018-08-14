# AutoEncoders

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

# Creating the architecture of the Neural Network


class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()  # Getting inherited classes and methods
        self._fc1 = nn.Linear(nb_movies, 20)  # 20 first hidden nodes
        self._fc2 = nn.Linear(20, 10)
        self._fc3 = nn.Linear(10, 20)  # Where the decoding starts
        self._fc4 = nn.Linear(20, nb_movies)
        self._activation = nn.Sigmoid()

    def forward(self, x):
        x = self._activation(self._fc1(x))
        x = self._activation(self._fc2(x))
        x = self._activation(self._fc3(x))
        x = self._fc4(x)
        return x

    def __repr__(self):
        return "%s (_fc1=%r, _fc2=%r, _fc3=%r, _fc4=%r, _activation=%r)" % \
                (self.__class__.__name__, self._fc1, self._fc2, self._fc3,
                 self._fc4, self._activation)


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
print("Initiating training ...")
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)

            # Not computing the gradiant with respect to the target
            # saving a lot of computations optimizing the code
            target.require_grad = False

            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / \
                float(torch.sum(target.data > 0) + 1e-10)  # To avoid div by 0
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
            optimizer.step()
    print(f"epoch: {epoch}/{nb_epoch} | loss: {train_loss/s}")

# Testing the SAE
print("Initiating testing ... ", end="", flush=True)
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)

        # Not computing the gradiant with respect to the target
        # saving a lot of computations optimizing the code
        target.require_grad = False

        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / \
            float(torch.sum(target.data > 0) + 1e-10)  # To avoid div by 0
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s += 1.
print("[DONE]")
print(f"loss: {test_loss/s}")

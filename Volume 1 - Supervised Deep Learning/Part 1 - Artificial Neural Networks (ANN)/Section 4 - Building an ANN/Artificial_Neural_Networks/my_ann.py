# Artificial Neural Network


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

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website:
# https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing


# Importing the libraries
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the Keras libraries and packages
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer (with dropout)
# Dropout disables a percentage (rate) of neurons to prevent overfitting.
# A to high percentage could cause underfitting.
# Nodes in the hidden layer = (Number of nodes in the input layer + 1)/2
# relu correspond to the rectifier function
classifier.add(Dense(activation="relu", input_dim=11,
                     units=6, kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
"""If you have more than two categories then the units and activation
parameter have to change, for example, units=3, activation="softmax"
"""
classifier.add(Dense(activation="sigmoid", units=1,
                     kernel_initializer="uniform"))

# Compiling the ANN
# adam correspond to a stochastic gradient descent algorithm
"""logaritmic loss function
   If dependent variable has a binary outcome then binary_crossentropy
   If dependent variable has more than 2 outcomes like 3 categories then
   the logaritmic loss function is called categorical_crossentropy
"""
classifier.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=25, epochs=500)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nEvaluating performance for ANN model")
eval_performance(cm)

# OPTIONAL: Save the model for future uses
# serialize model to JSON
model_json = classifier.to_json()
with open("ann_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("ann_model.h5")
print("Saved model to disk")

## OPTIONAL: Loading the saved model from disk
#json_file = open('ann_model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#new_classifier = model_from_json(loaded_model_json)
## load weights into new model
#new_classifier.load_weights('ann_model.h5')
#print("Loaded model from disk")
## Compiling the new model for it's previous use
#new_classifier.compile(
#    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Homework (my solution)

# Importing the homework dataset
hdataset = pd.read_csv('Homework.csv')
X_htest = hdataset.values

# Encoding categorical data
X_htest[:, 1] = labelencoder_X_1.transform(X_htest[:, 1])
X_htest[:, 2] = labelencoder_X_2.transform(X_htest[:, 2])
X_htest = onehotencoder.transform(X_htest).toarray()
X_htest = X_htest[:, 1:]

# Feature Scaling
X_htest = sc.transform(X_htest)

# Predicting the Test set results
y_hpred = classifier.predict(X_htest)
y_hpred = (y_hpred > 0.5)

# Homework (teacher's solution)
new_prediction = classifier.predict(
    sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Part 4 - Evaluation, Improving, and Tuning the ANN

# Evaluating the ANN


def build_classifier():
    """build_classifier Build Function for a ANN Classifier

    Build the classifier for an ANN

    Returns:
        Sequential -- ANN Classifier
    """

    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11,
                         units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6,
                         kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1,
                         kernel_initializer="uniform"))
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(
    build_fn=build_classifier, batch_size=25, epochs=500)

accuracies = cross_val_score(
    estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)

mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
# classifier.add(Dropout(rate=0.1))

# Tuning the ANN


def build_classifier_gs(optimizer):
    """build_classifier_gs Build Function for an ANN Classifier

    Build an ANN classifier for Grid Search

    Arguments:
        optimizer {str} -- The name of the algorithm to use

    Returns:
        Sequential -- The ANN Classifier
    """

    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11,
                         units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6,
                         kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1,
                         kernel_initializer="uniform"))
    classifier.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier_gs)
parameters = {
    'batch_size': [25, 32],
    'epochs': [100, 500],
    'optimizer': ['adam', 'rmsprop']
}
grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameters,
    scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X=X_train, y=y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website:
# https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN


import numpy as np
# Importing the Keras Libraries and packages
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                      input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a Second Convolutional Layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3),
                      activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='sigmoid', units=1))

# Compiling the CNN
classifier.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 = Fitting the CNN to the images
# Code from https://keras.io/preprocessing/image/


train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set', target_size=(64, 64),
    batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set', target_size=(64, 64),
    batch_size=32, class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=25,
                         validation_data=test_set, validation_steps=2000)

# OPTIONAL: Save the model for future uses
# serialize model to JSON
model_json = classifier.to_json()
with open("cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("cnn_model.h5")
print("Saved model to disk")

# OPTIONAL: Loading the saved model from disk
#json_file = open('cnn_model.json', 'r')
#loaded_model_json = json_file.read()
# json_file.close()
#new_classifier = model_from_json(loaded_model_json)
# load weights into new model
# new_classifier.load_weights('cnn_model.h5')
#print("Loaded model from disk")
# Compiling the new model for it's previous use
# new_classifier.compile(
#    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Homework (My solution)
# Couldn't test my solution, my PC is too slow...
#homework_datagen = ImageDataGenerator(rescale=1./255)
#
# homework_set = homework_datagen.flow_from_directory(
#    'dataset/single_prediction', target_size=(64, 64),
#    batch_size=32, class_mode='binary'
# )
#
#new_prediction = classifier.predict_generator(homework_set)

# Spoiler: This didn't do the job

# Homework (Teacher's solution)
# Part 3 - Making new predictions
test_image = image.load_img(
    'dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)  # Adds 3 dimentions
# 4th dimention for the batch
# axis is the position of the index of the dimension we are adding
# axis=0 means that the index of this new dimension is going to have the first
# index that is index zero
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'
print(f"The Prediction for the image 'cat_or_dog_2.jpg' was: {prediction}")

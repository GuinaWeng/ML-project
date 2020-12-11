"""

Team:
Guina WENG
Pin ZHU
Roma SURYAVANSHI
   
"""

from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD    
from keras.preprocessing import image
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def main():

    # convert txt file into csv
    trainDataset = pd.read_csv('/home/pinz/ML/Challenge-20201208/Challenge_train/Challenge_train/train.anno.txt', header = None)
    trainDataset.columns = ['file_name ' 'indoor ' 'outdoor ' 'person ' 'day ' 'night ' 'water ' 'road ' 'vegetation ' 'tree ' 'mountains ' 'beach ' 'buildings ' 'sky ' 'sunny ' 'partly_cloudy ' 'overcast ' 'animal ']
    trainDataset.head()
    testDataset = pd.read_csv('/home/pinz/ML/Challenge-20201208/Challenge_test/Challenge_test/test.anno.txt')
    testDataset.head()
    
    
    train_image = []
    
    for i in tqdm(range(trainDataset.shape[0])):
    imgTrain = tf.keras.preprocessing.image.load_img('/home/pinz/ML/Project2020-2021/train_data/' + trainDataset['file_name'][i] + '.jpg', target_size = None)
    imgTrain = image.img_to_array(imgTrain)
    imgTrain = img / 255
    train_image.append(imgTrain) 
    
    
    test_image = []
    
    for i in tqdm(range(trainDataset.shape[0])):
    imgTest = tf.keras.preprocessing.image.load_img('/home/pinz/ML/Challenge-20201208/Challenge_test/Challenge_test/test/' + testDataset['file_name'][i] + '.jpg', target_size = None)
    imgTest = image.img_to_array(imgTest)
    imgTest = img / 255
    test_image.append(imgTest) 
    
    (x_train, y_train) = train_image.load_data()
    (x_test, y_test) = test_image.load_data()
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of channels
    channels = 1
    # Total number of classes
    num_classes = 17
    
    # Normalize the images
    x_train = x_train / 255
    x_test = x_test / 255
    
    # convert labels to one-hot vectors
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)   
    
    # Choose only a subset of data for training
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    
    # Reshape the input to match the Keras's expectations (n_images, x_shape, y_shape, channels)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    
    # Print sizes of training and test datasets
    print('Size of training data:', x_train.shape, '\n') 
    print('Size of test data:', x_test.shape, '\n')     
    
    # Build CNN
    model = Sequential()
    # Add convolutional layer
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    # Add pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # Prepare data for fully connected neural network
    model.add(Flatten())
    # Fully connected neural network
    model.add(Dense(units=128, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.25))
    # Output layer
    model.add(Dense(units=num_classes, activation='softmax'))
    # Configure the model for training
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
    
    # Print model summary
    model.summary()
    
    # Number of epochs
    epochs = 10
    
    # Train the network
    model.fit(x_train, y_train,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_split=0.2)
    
    # Evaluate the model
    acc = model.evaluate(x_test, y_test, verbose=1)
    print('\nTest accuracy: %.2f%%' % (acc[1]*100))
       
    return x_train, x_test, y_train, y_test, acc
    
x_train, x_test, y_train, y_test, acc = main()

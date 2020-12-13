"""

Team:
Guina WENG
Pin ZHU
Roma SURYAVANSHI
   
"""


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    #Preparing Data and data lables
    trainDF = pd.read_csv('/home/pinz/ML/Challenge-20201208/Challenge_train/Challenge_train/train.anno.txt', delim_whitespace=True, header=None)
    testDF = pd.read_csv('/home/pinz/ML/Challenge-20201208/Challenge_test/Challenge_test/test.anno.txt', delim_whitespace=True, header=None, skiprows=[0])
    classes = np.array(['Indoor', 'Outdoor', 'Person', 'Day', 'Night', 'Water', 'Road', 
                        'Vegetation', 'Tree', 'Mountains', 'Beach', 'Buildings', 'Sky', 
                        'Sunny','Partly_Cloudy', 'Overcast', 'Animal'])

    #Preparing Training Data
    train_image = []
    for i in tqdm(range(trainDF.shape[0])):
        img = image.load_img('/home/pinz/ML/Challenge-20201208/Challenge_train/Challenge_train/train/'+trainDF[0][i],target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
    X_train = np.array(train_image)

    Y_train = np.array(trainDF.drop([0],axis=1))
    Y_train.shape

    #Preparing Training Data
    test_image = []
    for i in tqdm(range(testDF.shape[0])):
        img = image.load_img('/home/pinz/ML/Challenge-20201208/Challenge_test/Challenge_test/test/'+testDF[0][i],target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
    X_test = np.array(test_image)

    Y_test = np.array(testDF.drop([0],axis=1))
    Y_test.shape

    # Build CNN
    model = Sequential()
    
    # Add convolutional layer
    model.add(Conv2D(filters=50, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    
    # Add convolutional layer
    model.add(Conv2D(filters=50, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    
    # Add pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    # Dropout for regularization
    model.add(Dropout(0.25))
    
    # Add convolutional layer
    model.add(Conv2D(filters=125, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    
    # Add pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    # Dropout for regularization
    model.add(Dropout(0.25))
    
    # Prepare data for fully connected neural network
    model.add(Flatten())
    
    # Fully connected neural network
    model.add(Dense(units=500, activation='relu'))
    
    # Dropout for regularization
    model.add(Dropout(0.4))
    
    # Fully connected neural network
    model.add(Dense(units=250, activation='relu'))
    
    # Dropout for regularization
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(units=num_classes, activation='softmax'))
    
    # Configure the model for training
    #model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
    
    # Print model summary
    model.summary()

    #Compiling CNN Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Evaluaing Model
    model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test), batch_size=32)

    #Taking image to do validation of model prediction / once done we can pass whole teting data
    img = image.load_img('/home/pinz/ML/Challenge-20201208/Challenge_test/Challenge_test/test/27-27752.jpg',target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img/255
    plt.imshow(img)

    #Predicting values for image we have given as input
    proba = model.predict(img.reshape(1,224,224,3))

    #Display lables which has probability > 50%
    for i in range(len(classes)):   
        if proba[0][i]> 0.5 : 
            print("{}".format(classes[i])+" ({:.3})".format(proba[0][i]))
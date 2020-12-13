"""

Team:
Guina WENG
Pin ZHU
Roma SURYAVANSHI
   
"""


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications import VGG16
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    #Preparing Data and data lables
    trainDF = pd.read_csv('/home/pinz/ML/Challenge/Challenge_train/Challenge_train/train.anno.txt', delim_whitespace=True, header=None)
    testDF = pd.read_csv('/home/pinz/ML/Challenge/Challenge_test/Challenge_test/test.anno.txt', delim_whitespace=True, header=None, skiprows=[0])
    classes = np.array(['Indoor', 'Outdoor', 'Person', 'Day', 'Night', 'Water', 'Road', 
                        'Vegetation', 'Tree', 'Mountains', 'Beach', 'Buildings', 'Sky', 
                        'Sunny','Partly_Cloudy', 'Overcast', 'Animal'])

    #Preparing Training Data
    train_image = []
    for i in tqdm(range(trainDF.shape[0])):
        img = image.load_img('/home/pinz/ML/Challenge/Challenge_train/Challenge_train/train/'+trainDF[0][i],target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
    X_train = np.array(train_image)

    Y_train = np.array(trainDF.drop([0],axis=1))
    Y_train.shape

    #Preparing Training Data
    test_image = []
    for i in tqdm(range(testDF.shape[0])):
        img = image.load_img('/home/pinz/ML/Challenge/Challenge_test/Challenge_test/test/'+testDF[0][i],target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
    X_test = np.array(test_image)

    Y_test = np.array(testDF.drop([0],axis=1))
    Y_test.shape
    
    # Print sizes of training and test datasets
    print('Size of training data:', X_train.shape, '\n') 
    print('Size of test data:', X_test.shape, '\n') 

    # Build CNN
    model = Sequential()
    
    # Add convolutional layer
    model.add(Conv2D(filters=50, kernel_size=(5, 5), strides=1, padding='same', activation='relu', input_shape=(224, 224, 3)))
   
    # Add convolutional layer
    model.add(Conv2D(filters=50, kernel_size=(5, 5), strides=1, padding='same', activation='relu', input_shape=(224, 224, 3)))
    
    # Add pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    # Dropout for regularization
    model.add(Dropout(0.25))
    
    # Add convolutional layer
    model.add(Conv2D(filters=125, kernel_size=(5, 5), strides=1, padding='same', activation='relu', input_shape=(224, 224, 3)))
    
    # Add pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    # Dropout for regularization
    model.add(Dropout(0.25))
    
    # Prepare data for fully connected neural network
    model.add(Flatten())
    
    # Fully connected neural network
    model.add(Dense(units=250, activation='relu'))
    
    # Dropout for regularization
    model.add(Dropout(0.4))
    
    # Fully connected neural network
    model.add(Dense(units=128, activation='relu'))
    
    # Dropout for regularization
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(units=17, activation='softmax'))

    #Compiling CNN Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    model.summary()

    #Evaluaing Model
    model.fit(X_train, Y_train, 
              epochs=10, 
              validation_data=(X_test, Y_test),
              batch_size=32)

    #Taking 5 images to do validation of model prediction / once done we can pass whole teting data
    img1 = image.load_img('/home/pinz/ML/Challenge/Challenge_test/Challenge_test/test/27-27752.jpg',target_size=(224,224,3))
    
    img1 = image.img_to_array(img1)
    img1 = img1/255
    plt.imshow(img1)

    #Predicting values for image we have given as input
    proba1 = model.predict(img1.reshape(1,224,224,3))

    #Display lables which has probability > 50%
    for i in range(len(classes)):   
        if proba1[0][i]> 0.5 : 
            print("{}".format(classes[i])+" ({:.3})".format(proba1[0][i]))
    
    img2 = image.load_img('/home/pinz/ML/Challenge/Challenge_test/Challenge_test/test/27-27707.jpg',target_size=(224,224,3))
    
    img2 = image.img_to_array(img2)
    img2 = img2/255
    plt.imshow(img2)

    #Predicting values for image we have given as input
    proba2 = model.predict(img2.reshape(1,224,224,3))

    #Display lables which has probability > 50%
    for i in range(len(classes)):   
        if proba2[0][i]> 0.5 : 
            print("{}".format(classes[i])+" ({:.3})".format(proba2[0][i]))
    
    img3 = image.load_img('/home/pinz/ML/Challenge/Challenge_test/Challenge_test/test/27-27891.jpg',target_size=(224,224,3))
    
    img3 = image.img_to_array(img3)
    img3 = img3/255
    plt.imshow(img3)

    #Predicting values for image we have given as input
    proba3 = model.predict(img3.reshape(1,224,224,3))

    #Display lables which has probability > 50%
    for i in range(len(classes)):   
        if proba3[0][i]> 0.5 : 
            print("{}".format(classes[i])+" ({:.3})".format(proba3[0][i]))        
    
    img4 = image.load_img('/home/pinz/ML/Challenge/Challenge_test/Challenge_test/test/28-28175.jpg',target_size=(224,224,3))
    
    img4 = image.img_to_array(img4)
    img4 = img4/255
    plt.imshow(img4)

    #Predicting values for image we have given as input
    proba4 = model.predict(img4.reshape(1,224,224,3))

    #Display lables which has probability > 50%
    for i in range(len(classes)):   
        if proba4[0][i]> 0.5 : 
            print("{}".format(classes[i])+" ({:.3})".format(proba4[0][i]))
    
    img5 = image.load_img('/home/pinz/ML/Challenge/Challenge_test/Challenge_test/test/28-28058.jpg',target_size=(224,224,3))
    
    img5 = image.img_to_array(img5)
    img5 = img5/255
    plt.imshow(img5)

    #Predicting values for image we have given as input
    proba5 = model.predict(img5.reshape(1,224,224,3))

    #Display lables which has probability > 50%
    for i in range(len(classes)):   
        if proba5[0][i]> 0.5 : 
            print("{}".format(classes[i])+" ({:.3})".format(proba5[0][i]))
            
                   
x_train, x_test, y_train, y_test, acc = main()


# VGG16
pretrained_model = VGG16(include_top = False, weights = 'imagenet')
pretrained_model.summary()

vgg_feature_train = pretrained_model.predict(train_image)
vgg_feature_test = pretrained_model.predict(test_image)




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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Preparing Data and data lables
trainDF = pd.read_csv('train.anno.txt', delim_whitespace=True, header=None)
testDF = pd.read_csv('test.anno.txt', delim_whitespace=True, header=None, skiprows=[0])
classes = np.array(['Indoor', 'Outdoor', 'Person', 'Day', 'Night', 'Water', 'Road', 
                  'Vegetation', 'Tree', 'Mountains', 'Beach', 'Buildings', 'Sky', 
                  'Sunny','Partly_Cloudy', 'Overcast', 'Animal'])

#Preparing Training Data
train_image = []
for i in tqdm(range(trainDF.shape[0])):
    img = image.load_img('train/'+trainDF[0][i],target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X_train = np.array(train_image)

Y_train = np.array(trainDF.drop([0],axis=1))
print('Size of training data:', X_train.shape, '\n')

#Preparing Testing Data
test_image = []
for i in tqdm(range(testDF.shape[0])):
    img = image.load_img('test/'+testDF[0][i],target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
X_test = np.array(test_image)

Y_test = np.array(testDF.drop([0],axis=1))
print('Size of test data:', X_test.shape, '\n')

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Evaluaing Model
model.fit(X_train, Y_train, epochs=30, validation_split=0.3, batch_size=64)

print("Evaluate on test data")
results = model.evaluate(X_test, Y_test, batch_size=64)
print("test loss, test acc:", results)

pred = model.predict(X_test)
pred_rnd = pred.round()
metric = keras.metrics.Accuracy()
metric.update_state(Y_test,pred_rnd)
metric.result().numpy()

def tags_prediction(image_name):
  img1 = image.load_img('test/'+image_name,target_size=(224,224,3))
  img1 = image.img_to_array(img1)
  img1 = img1/255
  plt.imshow(img1)

  #Predicting values for image we have given as input
  prediction = model.predict(img1.reshape(1,224,224,3))
  values = prediction.round()
  
  # collect all predicted tags
  tags = [classes[i] for i in range(len(classes)) if values[0][i] == 1.0]
  return tags

tags_prediction(str(testDF[0][14]))

tags_prediction(str(testDF[0][144]))

tags_prediction(str(testDF[0][300]))

tags_prediction(str(testDF[0][923]))

tags_prediction(str(testDF[0][10]))

tags_prediction(str(testDF[0][8]))

tags_prediction(str(testDF[0][26])). #Model give some wrong prediction too as it is only 85% accurate

# Remove the first column (file name)
y_true =  Y_test #y_true.drop(testDF, axis=1)
y_pred = pred_rnd

# Calculate precision, recall, and f1 score
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

# Prepare for printing
dash = '-' * 45
# Loop over classes
for i in range(len(classes)+1):
    # Print the header
    if i == 0:
      print(dash)
      print('{:<15}{:<12}{:<9}{:<4}'.format('Class','precision','recall','f1 score'))
      print(dash)
    # Print precision, recall and f1 score for each of the labels   
    else:
      print('{:<17}{:<11.2f}{:<10.2f}{:<10.2f}'.format(classes[i-1],precision[i-1],recall[i-1],f1[i-1]))

# Print average precision     
precision_micro = precision_score(y_true, y_pred, average='micro')
print('{:<20}{:<4.2f}'.format('\nAverage precision:',precision_micro))
# Print average recall    
recall_micro = recall_score(y_true, y_pred, average='micro')
print('{:<19}{:<4.2f}'.format('Average recall:',recall_micro)) 
# Print average f1 score     
f1_micro = f1_score(y_true, y_pred, average='micro')
print('{:<19}{:<12.2f}'.format('Average f1 score:',f1_micro))


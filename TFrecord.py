import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
#from scipy import misc
import scipy.io as sio
import imageio
from PIL import Image
import os
 
train_path = r'/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train'
train_resize_path = r'/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train_data/'
test_path = r'/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_test/Challenge_test/test'
test_resize_path = r'/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_test/Challenge_test/test_data/'

#file_name indoor outdoor person day night water road vegetation tree mountains beach buildings sky sunny partly_cloudy overcast animal
def resize(file_path,resize_path):
    for i in os.listdir(file_path):
        img = Image.open(os.path.join(file_path,i))
        if img.mode != 'RGB':
           img = img.convert('RGB')
        out = img.resize((224, 224))
        if not os.path.exists(resize_path):
            os.makedirs(resize_path)
        out.save(os.path.join(resize_path, i))
    
train_data=resize(train_path,train_resize_path)
test_data=resize(test_path,test_resize_path)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))
 
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
 
 

train_txtfile = '/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train.anno.txt'
test_txtfile = '/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_test/Challenge_test/test.anno.txt' 
train_tfrecords_filename= '/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train_data.tfrecord'
test_tfrecords_filename = '/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_test/Challenge_test/test_data.tfrecord'



def write_tfrecord(resize_path,txtfile,tfrecords_filename):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    file_lines = open(txtfile).readlines()
    for idx,line in enumerate(file_lines):
        line = line.strip('\n')
        imgname=line.split()[0]
        img = np.float64(imageio.imread(resize_path + imgname))
        label = line.split()[1:]
        label = "".join(label)
        label = int(label)
        img_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw),
            'label': _int64_feature(label)}))
 
        writer.write(example.SerializeToString())
    writer.close() 
   
train_tfrecords = write_tfrecord(train_resize_path,train_txtfile,train_tfrecords_filename)
test_tfrecords = write_tfrecord(test_resize_path,test_txtfile,test_tfrecords_filename)

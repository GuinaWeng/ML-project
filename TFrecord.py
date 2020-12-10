import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
#from scipy import misc
import scipy.io as sio
import imageio
from PIL import Image
import os
 
img_path = r'/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train'
resize_path = r'/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train_data/'


    
for i in os.listdir(img_path):
    img = Image.open(os.path.join(img_path,i))
    if img.mode != 'RGB':
       img = img.convert('RGB')
    out = img.resize((224, 224))
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)
    out.save(os.path.join(resize_path, i)) 
 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))
 
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
 
 
tfrecords_filename = "/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train_data.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
 

txtfile = '/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train.anno.txt'

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

import tensorflow as tf
import os

train_tfrecords_filename= '/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_train/Challenge_train/train_data.tfrecord'
test_tfrecords_filename = '/Users/nana/Desktop/CNN/Challenge-20201201/Challenge_test/Challenge_test/test_data.tfrecord'

def decoder(tfrecord_file, is_train_dataset=None):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    feature_discription = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
 
    def _parse_example(example_string): # 解码每一个example
        feature_dic = tf.io.parse_single_example(example_string, feature_discription)
        feature_dic['image_raw'] = tf.io.decode_jpeg(feature_dic['image_raw'])
        return feature_dic['image_raw'], feature_dic['label']
 
    batch_size = 32
 
    if is_train_dataset is not None:
        dataset = dataset.map(_parse_example).shuffle(buffer_size=2000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(_parse_example)
        dataset = dataset.batch(batch_size)
 
    return dataset
 
train_data = decoder(train_tfrecords_filename, 1)
test_data = decoder(test_tfrecords_filename)


#coding = utf8

import numpy as np
import tensorflow as tf
import os, time
from PIL import Image

def get_filenames(file_dir):
    '''return filenames and labels'''
    for root, dirs, files in os.walk(file_dir):
        print(root, dirs)
        labels = [int(fn.replace('.jpg', '').split('_')[1]) for fn in files]
        assert(len(files) != 0)
        return files, labels

def write_to_tfrecords(file_dir, record_dir, record_name):
    writer = tf.python_io.TFRecordWriter(record_dir + '/' + record_name)
    filenames, labels = get_filenames(file_dir)
    count = 0
    for imagename, label in zip(filenames, labels):
         img_path = file_dir + '/' + imagename
         img = Image.open(img_path)
         img_raw = img.tobytes()
         feature_label = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
         feature_img = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
         example = tf.train.Example(features=tf.train.Features(feature={'label':feature_label, 'img_raw':feature_img}))
         writer.write(example.SerializeToString())
         count += 1
         if count % 1000 == 0:
             print('write %d images'%(count))
    writer.close()

if __name__ == '__main__':
    file_dir = '/Users/david/Desktop/tensorflow_exercise/data/cifar/jpg_test'
    record_dir = '.'
    write_to_tfrecords(file_dir, record_dir, 'cifar_test.tfrecords')

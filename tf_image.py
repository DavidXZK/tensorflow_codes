#coding=utf8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def image_augmentation(filename):
    '''use tf.image'''
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    print(image_decoded.shape)
    #--------------image crop--------------------
    image_distorted = tf.image.resize(image_decoded, size=(350,350))
    #image_distorted = tf.image.resize_image_with_crop_or_pad(image_decoded, 500, 500)
    image_distorted = tf.random_crop(image_distorted, size=(300, 300, 3))
    #-------------flip and whiten-----------------
    image_distorted = tf.image.random_flip_left_right(image_distorted)
    image_distorted = tf.image.per_image_standardization(image_distorted)
    return image_distorted


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.ERROR)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        image_distorted = tf.cast(image_augmentation('rockets.jpg'), tf.uint8)
        plt.title('image_distorted')
        plt.imshow(sess.run(image_distorted))
        plt.show()
        

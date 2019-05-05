#coding=utf8
'''use tf.dataset api and tf api to build a cnn'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def _parse_func(example_proto):
    '''parse example'''
    example_feature = {'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(example_proto, features=example_feature)
    label = features['label']
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, shape=(32,32,3))
    img = tf.cast(img, tf.float32) / 255.
    label = tf.cast(label, tf.int32)
    return img, label

def input_fn(filepath, batch_size, training=True):
    '''get input pipline'''
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_func, num_parallel_calls=4)
    if training:
        dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(batch_size=batch_size).prefetch(2 * batch_size)
    return dataset

def initialize_parameters(conv_dims):
    ''''''
    parameters = {}
    nums = len(conv_dims)
    for i in range(nums):
        w_name, b_name = 'w'+str(i+1), 'b'+str(i+1)
        parameters[w_name] = tf.get_variable(w_name, shape=conv_dims[i], initializer=tf.contrib.layers.xavier_initializer())
        parameters[b_name] = tf.get_variable(b_name, shape=(conv_dims[i][-1]), initializer=tf.constant_initializer(0.0))
    return parameters

def create_placeholder(name, shape, dtype=tf.float32):
    return tf.placeholder(dtype, shape=shape, name=name)

def conv_forward(X, parameters):
    '''m*h*w*c->conv*2->max_pool->conv*2->max_pool->flatten->dense(128)->dense(64)->dense(32)->softmax(10)'''
    Z = tf.nn.conv2d(X, filter=parameters['w1'], strides=[1,1,1,1], padding='SAME', name='conv1') + parameters['b1']
    A = tf.nn.relu(Z, name='relu1')
    Z = tf.nn.conv2d(A, filter=parameters['w2'], strides=[1,1,1,1], padding='SAME', name='conv2') + parameters['b2']
    A = tf.nn.relu(Z, name='relu2')
    A_pooling = tf.nn.max_pool(A, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='max_pool1')
    Z = tf.nn.conv2d(A_pooling, filter=parameters['w3'], strides=[1,1,1,1], padding='SAME', name='conv3') + parameters['b3']
    A = tf.nn.relu(Z, name='relu3')
    Z = tf.nn.conv2d(A, filter=parameters['w4'], strides=[1,1,1,1], padding='SAME', name='conv4') + parameters['b4']
    A = tf.nn.relu(Z, name='relu4')
    A_pooling = tf.nn.max_pool(A, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='max_pool2')
    flatten = tf.layers.flatten(A_pooling, name='flatten_input')
    A1 = tf.layers.dense(flatten, units=128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense1')
    A2 = tf.layers.dense(A1, units=64, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
    A3 = tf.layers.dense(A2, units=32, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense3')
    logits = tf.layers.dense(A3, units=10, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Z4')
    return logits #(batch, depth)

def compute_cost(logits, Y):
    return tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)

def model(conv_dims, params, epochs=100, print_cost=True):
    tf.reset_default_graph()
    costs = []
    num_batch = np.ceil(params['data_size'] / params['batch_size']) #num of batches
    print('num_batch:', num_batch)
    parameters = initialize_parameters(conv_dims)
    dataset = input_fn(params['filepath'], params['batch_size'])
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    X, Y = create_placeholder('X', shape=[None,32,32,3]), create_placeholder('Y', shape=[None, ], dtype=tf.int32)
    Y_one_hot = tf.one_hot(Y, 10, axis=-1) #(batch, depth)
    #print(Y_one_hot.shape)
    logits = conv_forward(X, parameters)
    loss_op = compute_cost(logits, Y_one_hot)
    train_op = tf.train.AdamOptimizer(params['lr']).minimize(loss_op)
    acc_op = tf.metrics.accuracy(labels=tf.argmax(Y_one_hot), predictions=tf.argmax(logits))
    #------------------------------------------------------------------------------------------
    dataset_test = input_fn(params['testpath'], params['batch_size'], training=False)
    iterator_test = dataset_test.make_initializable_iterator()
    next_batch_test = iterator_test.get_next()
    #------------------------------------------------------------------------------------------
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_local)
        for i in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            sess.run(iterator.initializer)
            index = 0
            start = time.process_time()
            while True:
                batch_loss, batch_acc = 0, 0
                try:
                    batch_x, batch_y = sess.run(next_batch)
                    #print('batch_x:', batch_x.shape)
                    batch_loss, (batch_acc, _), _ = sess.run([loss_op, acc_op, train_op], feed_dict={X: batch_x, Y: batch_y})
                    if index%100 == 0:
                        print(batch_loss, end=' ')
                    index += 1
                    #print('batch_acc: ', batch_acc)
                except tf.errors.OutOfRangeError:
                    break
                epoch_loss += batch_loss
                epoch_acc += batch_acc
            epoch_loss /= num_batch
            epoch_acc /= num_batch
            print()
            end = time.process_time()
            if print_cost:
                costs.append(epoch_loss)
                print('%dth epoch cost = %f, acc = %f, time = %fs'%(i, epoch_loss, epoch_acc, end-start))
        #--------------------------------------------------------------------------
        parameters = sess.run(parameters)
        epoch_acc_test = 0
        num_batch_test = np.ceil(params['test_size'] / params['batch_size'])
        sess.run(iterator_test.initializer)
        while True:
            try:
                batch_x, batch_y = sess.run(next_batch_test)
                batch_loss, (batch_acc, _) = sess.run([loss_op, acc_op], feed_dict={X: batch_x, Y: batch_y})
            except tf.errors.OutOfRangeError:
                break
            epoch_acc_test += batch_acc
        epoch_acc_test /= num_batch_test
        print('test set cost = %f'%(epoch_acc_test))

    return parameters, costs

def plot_accuracy(lr, costs):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('lr = %f'%(lr))
    x = np.range(len(costs))
    plt.plot(x, costs)

def predict(X, parameters):
    tf.reset_default_graph()
    init = tf.global_variables_initializer()
    logits = conv_forward(X, parameters)
    labels = tf.argmax(logits)
    with tf.Session() as sess:
        sess.run(init)
        labels = sess.run(labels)
        return labels

def test():
    a = tf.constant([0,2,2,3,3]) #(batch, depth)
    b = tf.constant([2,2,2,2,3])
    a_one_hot = tf.one_hot(a, 4)
    b_one_hot = tf.one_hot(b, 4) #(batch, depth)
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=a_one_hot, logits=b_one_hot, reduction='none')
    print(a_one_hot.shape)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(loss_op))

if __name__ == '__main__':
    params = {'filepath': 'cifar_train.tfrecords', 
              'testpath': 'cifar_test.tfrecords',
              'data_size': 50000, 
              'test_size': 10000, 
              'lr': 0.005, 
              'batch_size': 64}
    conv_dims = [[3,3,3,32],[3,3,32,64],[3,3,64,128],[3,3,128,256]]
    start = time.process_time()
    parameters, costs = model(conv_dims, params, epochs=100)
    #test()
    end = time.process_time()
    print('time consume: %ds'%(end - start))

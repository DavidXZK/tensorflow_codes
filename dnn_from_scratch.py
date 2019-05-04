#coding = utf8

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time, sys
import matplotlib.pyplot as plt

def load_data():
    '''load mnist from tensorflow'''
    '''60000*28*28, 10000*28*28, 0-255'''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T
    return x_train, y_train, x_test, y_test

def show_many_images(x, y):
    plt.figure(figsize = (10, 8))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap = plt.cm.get_cmap('binary'))
        plt.xlabel('label = %d'%(y[i]))
    plt.show()

def convert_to_one_hot(Y, C):
    '''C = the number of classes'''
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def mini_batches(X, Y, batch_size):
    batches = []
    m = X.shape[1]
    indexes = np.random.permutation(m)
    blens = m // batch_size
    for i in range(blens):
        batch_x = X[:, indexes[i*batch_size : (i+1)*batch_size]]
        batch_y = Y[:, indexes[i*batch_size : (i+1)*batch_size]]
        batches.append((batch_x, batch_y))
    if blens * batch_size == m:
        return batches
    else:
        batch_x = X[:, indexes[blens*batch_size: ]]
        batch_y = Y[:, indexes[blens*batch_size: ]]
        batches.append((batch_x, batch_y))
    return batches

def activation(z, names = 'sigmoid'):
    if names == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif names == 'tanh':
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    elif names == 'relu':
        return np.maximum(z, 0)
    else:
        print('no activation named %s'%(names))
        sys.exit(-1)

def softmax(z):
    t = np.exp(z)
    return t / np.sum(t, axis = 0, keepdims=True)

def initialize_parameters(dims):
    '''initialize parameters w ,b'''
    # xavier initialization
    parameters = {}
    for i in range(len(dims)-1):
        parameters['w'+str(i+1)] = np.random.randn(dims[i+1], dims[i]) * np.sqrt(1 / dims[i])
        parameters['b'+str(i+1)] = np.zeros((dims[i+1], 1))

    return parameters

def cost_function(A, y):
    '''compute the cost'''
    m = y.shape[1]
    cost = - (np.sum(np.log(A) * y) / m)
    return np.squeeze(cost)

def activation_grads(A, name='relu'):
    if name == 'sigmoid':
        return A * (1 - A)
    elif name == 'tanh':
        return (1 - A**2)
    elif name == 'relu':
        return np.where(A > 0, 1, 0)

def forward_propagation(X, Y, parameters):
    '''compute the cost, cache a'''
    cache = [X]
    A = X
    l = len(parameters) // 2
    for i in range(1, l):
        w, b = parameters['w' + str(i)], parameters['b' + str(i)]
        Z = np.dot(w, A) + b # shape n(l+1), m
        A = activation(Z, 'relu')
        cache.append(A)
    w, b = parameters['w'+str(l)], parameters['b'+str(l)]
    Z = np.dot(w, A) + b
    A = softmax(Z)
    cost = cost_function(A, Y)
    dZ = A - Y
    return cache, cost, dZ

def backward_propagation(parameters, cache, dZ):
    '''compute and return grads'''
    grads = {}
    l = len(parameters) // 2
    m = dZ.shape[1]
    A = cache[-1]
    grads['dw'+str(l)] = np.dot(dZ, A.T) / m
    grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
    dA = np.dot(parameters['w' + str(l)].T, dZ)
    for i in range(l-1, 0, -1):
        dZ = dA * activation_grads(A, 'relu')
        A = cache[i - 1]
        grads['dw'+str(i)] = np.dot(dZ, A.T) / m
        grads['db'+str(i)] = np.sum(dZ, axis = 1, keepdims=True) / m
        dA = np.dot(parameters['w' + str(i)].T, dZ)
    return grads

def update_parameters(parameters, grads, lr):
    l = len(parameters) // 2
    for i in range(1, l+1):
        parameters['w'+str(i)] -= lr * grads['dw'+str(i)]
        parameters['b'+str(i)] -= lr * grads['db'+str(i)]

def predict(X, parameters):
    l = len(parameters) // 2
    A = X
    for i in range(1, l+1):
        w, b = parameters['w'+str(i)], parameters['b'+str(i)]
        Z = np.dot(w, A) + b
        A = activation(Z, names = 'relu')
    A = softmax(Z)
    return A

def dnn_model(datasets, dims, epochs = 1000, batch_size = 128, lr = 0.001, prints = True):
    x_train, y_train, x_test, y_test = datasets
    parameters = initialize_parameters(dims)
    costs = []
    for i in range(epochs):
        cost = 0
        batches = mini_batches(x_train, y_train, batch_size)
        for batch in batches:
            batch_x, batch_y = batch
            cache, batch_cost, dZ = forward_propagation(batch_x, batch_y, parameters)
            cost += batch_cost
            grads = backward_propagation(parameters, cache, dZ)
            update_parameters(parameters, grads, lr)
        cost = cost / len(batches)
        costs.append(cost)
        if prints:
            print('%dth epochs cost = %f'%(i, cost))
    y_hat_train = predict(x_train, parameters)
    prediction = np.sum(np.argmax(y_hat_train, axis = 0) == np.argmax(y_train, axis = 0)) / y_hat_train.shape[1]
    print('train_set percision is %f'%(prediction))
    y_hat_test = predict(x_test, parameters)
    prediction_test = np.sum(np.argmax(y_hat_test, axis=0) == np.argmax(y_test, axis=0)) / y_hat_test.shape[1]
    print('test_set percision is %f' % (prediction_test))
    return parameters

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    y_train, y_test = convert_to_one_hot(y_train, 10), convert_to_one_hot(y_test, 10)
    dataset = (x_train / 255, y_train, x_test / 255, y_test)
    dims = [784, 256, 128, 32, 10]
    start = time.process_time()
    parameters = dnn_model(dataset, dims, epochs = 300, batch_size=128, lr = 0.005)
    end = time.process_time()
    print('time comsume: %ds'%(end - start))

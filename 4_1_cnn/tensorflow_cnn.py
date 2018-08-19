import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
# from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])

    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)
    # 卷积层1的参数，卷积核的大小为4*4，3通道，8个卷积核
    W1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # 卷积层2的参数，卷积核大小未2*2， 8通道，16个卷积核
    W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


# 执行前向传播
def forward_propagation(X, parameters):
    # CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    W1 = parameters['W1']/ np.sqrt(2)
    W2 = parameters['W2']/ np.sqrt(2)
    # 卷积，步长为1
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # 非线性relu
    A1 = tf.nn.relu(Z1)
    # SAME池化，池化的窗口大小为8*8， 步长为8
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # 卷积，步长为1
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # 非线性relu
    A2 = tf.nn.relu(Z2)
    # SAME池化，池化出口大小为4*4， 步长为4
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # 将得到的特征给平坦化，得到的结果shape=(batch_size, k)
    P2 = tf.contrib.layers.flatten(P2)
    # 全连接层，该层的参数不需要进行初始化设定的， 6代表输出的结果的种类，activation_fn=None 表示不使用非线性激励函数，方便计算损失函数
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

    return Z3


# 利用交叉熵损失函数来计算损失，reduce_mean计算的是每个batch的平均cost
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True):
    # ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches  # 所有的batch的损失函数的和，除以batch的数量，得到所有原本的损失均值

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # 计算精确度
        correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))  # 因为使用了one-hot 所以要计算每个列的最大值所在的位置表示预测的结果
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))   # 将对象的值转化为float型，便且计算所有样本的平均

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train = X_train_orig / 255.  # (1080, 64, 64, 3)
    X_test = X_test_orig / 255.  # (120, 64, 64, 3)
    Y_train = convert_to_one_hot(Y_train_orig, 6).T  # (1080, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6).T  # (120, 6)

    _, _, parameters = model(X_train, Y_train, X_test, Y_test)


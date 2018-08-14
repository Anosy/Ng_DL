# 本代码简单粗略用tensorflow来搭建个单层的神经网络
import tensorflow as tf
from sklearn.metrics import accuracy_score
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import matplotlib.pyplot as plt

x, y = load_planar_dataset()
x, y = x.T, y.T
# 定义输入输出
X = tf.placeholder(tf.float32, [None, 2], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')
# 定义参数
w1 = tf.Variable(tf.random_normal([2, 4]), name='w1')
b1 = tf.Variable(tf.zeros([4]), name='b1')
w2 = tf.Variable(tf.random_normal([4, 1]), name='w2')
b2 = tf.Variable(tf.zeros([1]), name='b2')
# 前向传播
layer1_out = tf.nn.tanh(tf.matmul(X, w1) + b1)
# layer2_out = tf.nn.sigmoid(tf.matmul(layer1_out, w2) + b2)
layer2_out = tf.matmul(layer1_out, w2) + b2
# 计算损失函数
# cost = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(layer2_out, 1e-10, 1.0)))
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer2_out, labels=Y))
# 计算精确度
correction = tf.equal(tf.nn.sigmoid(layer2_out) > 0.5, Y > 0.5)
acc = tf.reduce_mean(tf.cast(correction, tf.float32))
# 定义优化方法
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1, 20001):
        _, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})

        if i % 2000 == 0:
            print('迭代%d次，损失值降低为%g' % (i, c))
    out = sess.run(layer2_out, feed_dict={X: x, Y: y})
    accuracy = sess.run(acc, feed_dict={X: x, Y: y})
    print('训练集的精确度为%g%%' % (accuracy * 100))

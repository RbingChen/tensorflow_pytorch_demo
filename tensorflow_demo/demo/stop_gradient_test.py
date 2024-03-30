# coding:utf-8
import tensorflow as tf

w1 = tf.Variable(2.0)
w2 = tf.Variable(88.)
w3 = tf.Variable(66.)

a = tf.multiply(w1, w2)
a_stop = tf.stop_gradient(a)

b = tf.multiply(w2, 3.)
c = tf.add(w3, a)
c_stop = tf.stop_gradient(c)

loss = tf.add(c,b)
gradients = tf.gradients(loss,[w1,w2,w3,a,b,c])
print(gradients)


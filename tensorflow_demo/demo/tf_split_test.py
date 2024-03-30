# coding:utf-8

import tensorflow as tf

sim = tf.random.uniform(shape=(8,8))

t = tf.split(sim,3)
with tf.Session() as sess:
    print(sess.run(t))
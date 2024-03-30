# coding:utf-8

import tensorflow as tf
import numpy as np
labels = tf.constant([[1],[2],[3]])
label2 = tf.constant([[5],[2],[3]])

cond1 = tf.where(tf.math.logical_and(labels*2>1,label2*2>5),labels,label2)
cond2 = tf.where(label2*2>5,labels,label2)
#cond2 = tf.where(cond1 )
y = tf.constant([[1],[0],[1],[0],[0],[1]])
y2 = tf.expand_dims(y, -1)
yT = tf.transpose(y2)
mask = tf.cast(tf.equal(y2, yT), tf.float32)
mask2 = tf.one_hot(tf.range(5),10)

nr = tf.constant([[1.0,2,3],[4.0,5,6]])
nr1 = tf.nn.l2_normalize(nr,0)
nr2 = tf.nn.l2_normalize(nr,1)
with tf.Session() as sess:
    #print(sess.run(cond1))
    #print(sess.run(cond2))
    #print(sess.run(mask))
    #print(sess.run(y2))
    #print(sess.run(yT))
    #print(sess.run(mask2))
    #print(sess.run(tt))
    print(sess.run(nr1))
    print(sess.run(nr2))
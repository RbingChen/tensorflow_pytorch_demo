# coding:utf-8

import tensorflow as tf
import numpy as np


emb = tf.random.uniform(shape=(8,4))

emb_l2 = tf.nn.l2_normalize(emb,axis=1)
emb_sim = tf.matmul(emb_l2,emb_l2,transpose_b=True)

pair_ones = tf.ones_like(emb_sim)
zero_ones = tf.zeros_like(emb_sim)
cond_one = tf.where(emb_sim>0.95,pair_ones,zero_ones)
cond_zero = tf.where(emb_sim<=0.95,pair_ones,zero_ones)
cond_one_sum = tf.reduce_sum(cond_one,axis=1,keepdims=True)
cond_zero_sum = tf.reduce_sum(cond_zero,axis=1,keepdims=True)
z = tf.zeros([2,3])
with tf.Session() as sess:
      print(sess.run(emb_sim))
      print(sess.run(cond_one_sum))
      print(sess.run(cond_zero_sum))
      print(sess.run(z))


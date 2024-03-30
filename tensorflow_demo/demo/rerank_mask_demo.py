# coding:utf-8

import tensorflow as tf
import numpy as np

dense_input = tf.constant([[1.0,2.3,1],
                           [1.0,2.3,2],
                           [1.0,2.3,3],
                           [1.0,2.3,1],
                           [1.0,2.3,5],
                           [1.0,2.3,4],
                           [1.0,2.3,2]
                           ])

buz_kind_size = tf.constant(5,tf.float32)
mix_buz_index = tf.gather(dense_input, [2], axis=1)
tag = mix_buz_index
mix_buz_index = tf.reduce_sum(mix_buz_index, axis=1)
mix_buz_index = buz_kind_size - mix_buz_index
zeros = tf.zeros_like(mix_buz_index)
mix_buz_index = tf.where(mix_buz_index < buz_kind_size, x=mix_buz_index, y=zeros)
mix_buz_index = tf.cast(mix_buz_index, dtype=tf.int32)
batch_size = tf.shape(mix_buz_index)[0]
mix_batch_index = tf.range(batch_size)
mix_index = tf.stack([mix_batch_index, mix_buz_index], axis=1)

mix_mask_base = tf.zeros([batch_size, 5], dtype=tf.int32)
mix_mask_ones = tf.ones([batch_size], dtype=tf.int32)
x_shape = tf.shape(mix_mask_base)
mix_mask = tf.scatter_nd(mix_index, mix_mask_ones, x_shape)
mix_mask = tf.expand_dims(mix_mask, axis=-1)
mix_mask = tf.cast(mix_mask, tf.float32)

with tf.Session() as sess:
    print(sess.run(mix_index))
    print(sess.run(batch_size))
    print(sess.run(mix_mask))

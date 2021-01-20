#!/usr/bin/env python3

"""
@author: Qingxin Meng
@since: 2018-08-02
"""

import photinia as ph
from module.utils import *


class AttentionLayer(ph.Widget):
    def __init__(self, name, output_dim):
        self.output_dim = output_dim
        super(AttentionLayer, self).__init__(name)

    def _build(self):
        self.a = tf.Variable(
            tf.random_uniform([self.output_dim], minval=1, maxval=10),
            dtype=tf.float32,
        )
        self.sigma = tf.Variable(
            tf.random_uniform([self.output_dim], minval=1, maxval=10),
            dtype=tf.float32,
        )

    def _setup(self, time_interval, index, mask, states):
        attn_batch = tf.map_fn(
            fn=lambda x: self.attn_for_one_elem(x[0], x[1], x[2], x[3]),
            elems=(time_interval, index, mask, states),
            dtype=tf.float32
        )

        return attn_batch

    def setup_sequence(self, time_interval, index, mask, states):
        time_span = tf.transpose(time_interval, (1, 0, 2))
        time_index = tf.transpose(index, (1, 0))
        attn_seq = tf.map_fn(
            fn=lambda x: self._setup(x[0], x[1], mask, states),
            elems=(time_span, time_index),
            dtype=tf.float32
        )
        return attn_seq

    def attn_for_one_elem(self, time_interval, index, mask, states):
        sigma = tf.nn.relu(self.sigma)[None, :]  # [1,latent_dim]
        temp = time_interval[:index][:, None]  # [seq,1]
        mask = mask[:index]
        mul = tf.exp(-(sigma * temp))  # [seq,latent_dim]
        alpha = self.a[None, :] * mul  # [seq,latent_dim]
        alpha_ = tf.boolean_mask(alpha, mask)
        states_ = tf.boolean_mask(states[:index], mask)
        attn = tf.reduce_sum(tf.nn.softmax(alpha_, 0) * states_,0)
        return attn

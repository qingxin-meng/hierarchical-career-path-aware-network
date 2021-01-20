#!/usr/bin/env python3

"""
@author: Qingxin Meng
@since: 2018-09-19
"""
# import sys
# sys.path.extend(['/home/mqx/time_sequence_prediction',
#                  '/home/mqx/time_sequence_prediction/module'])
import photinia as ph
import time
import random
from module.Data1207 import Data
from module.utils import *
from module.Evaluate import evaluate

# from tensorflow.python import debug as tf_debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Career(ph.Model):

    def __init__(self,
                 name,
                 company_count=None,
                 max_time=None,
                 emb_size=100,
                 emb_time_size=10,
                 emb_in_size=50,
                 emb_out_size=10,
                 latent_dim=200,
                 keep_prob=0.8):
        self._company_count = company_count
        self._max_time = max_time
        self._emb_size = emb_size
        self._emb_time_size = emb_time_size
        self._emb_in_size = emb_in_size
        self._emb_out_size = emb_out_size
        self._latent_dim = latent_dim
        self._keep_prob = keep_prob
        ph.Model.__init__(self, name)

    def _build(self):
        # 网络模块定义 --- build
        # self.graph=tf.Graph()
        # with self.graph.as_default():
        self._emb = ph.Linear('EMB', self._company_count, self._emb_size)
        self._emb_time = ph.Linear('EMB_TIME', self._max_time, self._emb_time_size)
        self._emb_in = ph.Linear('EMB_in', self._company_count, self._emb_in_size)
        self._emb_out = ph.Linear('EMB_out', 2, self._emb_out_size)
        self._LSTM = ph.LSTMCell('LSTM', self._emb_size + self._emb_time_size + self._emb_in_size,
                                 self._latent_dim)
        self._dense = ph.Linear('Dense', self._latent_dim, self._company_count)
        self._dense_lambda = ph.Linear('Dense_time', self._latent_dim + self._emb_out_size, self._max_time)

        # 输入定义

        self.seq_event = tf.placeholder(shape=(None, None,), dtype=tf.int32)
        self.seq_time = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.mask = tf.placeholder(shape=(None, None), dtype=tf.float32)
        self.target_event = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.target_time = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.seq_company_in = tf.placeholder(shape=(None, None, self._company_count), dtype=tf.float32)
        self.seq_company_out = tf.placeholder(shape=(None, None, 2), dtype=tf.float32)
        self.seq_duration_prob = tf.placeholder(shape=(None, None, self._max_time), dtype=tf.float32)

        self.batch_size = tf.shape(self.seq_event)[0]
        self.seq_length = tf.shape(self.seq_event)[1]

        seq_event_onehot = tf.one_hot(self.seq_event, depth=self._company_count) * self.mask[:, :, None]
        seq_time_onehot = tf.one_hot(self.seq_time, depth=self._max_time) * self.mask[:, :, None]

        seq_emb_event = tf.map_fn(fn=self.emb_process, elems=seq_event_onehot)
        seq_emb_time = tf.map_fn(fn=self.emb_time_process, elems=seq_time_onehot)
        seq_emb_in = tf.map_fn(fn=self.emb_in_process, elems=self.seq_company_in)
        seq_emb_out = tf.map_fn(fn=self.emb_out_process, elems=self.seq_company_out)  # [batch,seq,emb_out_size]

        seq_emb = tf.concat([seq_emb_event, seq_emb_time, seq_emb_in], -1)

        states = self._LSTM.setup_sequence(seq_emb)
        states = tf.nn.dropout(states, keep_prob=self._keep_prob)
        states_concat=tf.concat([states,seq_emb_out],-1)

        prob_next_event = tf.map_fn(
            fn=self.density_process,
            elems=states
        ) * self.mask[:, :, None]

        target_event_onehot = tf.one_hot(self.target_event, depth=self._company_count) * self.mask[:, :, None]

        loglike_event = tf.reduce_sum(tf.reduce_sum(tf.log(1e-9 + prob_next_event) * target_event_onehot, -1), -1)
        self.loglike_event = tf.reduce_mean(loglike_event)

        # next is train/predict survival time

        delta_lambda = tf.map_fn(
            fn=self.density_time_process,
            elems=states_concat
        ) * self.mask[:, :, None]  # [batch,seq,time_count]


        target_time_onehot = tf.one_hot(self.target_time, self._max_time) * self.mask[:, :, None]

        term1 = tf.log(1e-9 + tf.reduce_sum(delta_lambda * target_time_onehot, -1))  # [batch,seq]

        cum_delta_lambda = tf.reshape(tf.cumsum(delta_lambda, -1),
                                      [self.batch_size * self.seq_length, -1])  # [batch*seq,time_count]

        temp_seq = tf.reshape(self.target_time, [-1, 1])  # [batch*seq,1]

        cum_index = tf.range(self.batch_size * self.seq_length)[:, None]

        temp_index = tf.concat([cum_index, temp_seq], 1)

        term2 = tf.reshape(tf.gather_nd(cum_delta_lambda, temp_index),
                           [self.batch_size, -1]) * self.mask  # [batch*seq]

        self.loglike_duration = tf.reduce_mean(tf.reduce_sum((term1 - term2), -1))

        self.loss = -(self.loglike_event + self.loglike_duration)

        optimizer = tf.train.AdamOptimizer(1e-3)

        gradients = optimizer.compute_gradients(self.loss, tf.trainable_variables())

        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]

        self.train_op = optimizer.apply_gradients(capped_gradients)

        ## next is for validation predicting

        self._add_slot(
            'train',
            outputs=(self.loss,
                     self.loglike_event,
                     self.loglike_duration,
                     delta_lambda,
                     prob_next_event),
            inputs=(self.seq_event,
                    self.seq_time,
                    self.mask,
                    self.target_event,
                    self.target_time,
                    self.seq_company_in,
                    self.seq_company_out,
                    self.seq_duration_prob),
            updates=self.train_op
        )

        self._add_slot(
            'predict',
            outputs=(delta_lambda,
                     prob_next_event),
            inputs=(self.seq_event,
                    self.seq_time,
                    self.mask,
                    self.target_event,
                    self.target_time,
                    self.seq_company_in,
                    self.seq_company_out,
                    self.seq_duration_prob)
        )

    def emb_process(self, event):
        emb = self._emb.setup(event)
        return emb

    def emb_time_process(self, time):
        emb_time = self._emb_time.setup(time)
        return emb_time

    def emb_in_process(self, seq_in):
        emb_in = self._emb_in.setup(seq_in)
        return emb_in

    def emb_out_process(self, out):
        emb_out = self._emb_out(out)
        return emb_out

    def density_process(self, state):
        prob = self._dense.setup(state)
        prob = tf.nn.softmax(prob, -1)
        return prob

    def density_time_process(self, state):
        delta_lambda = self._dense_lambda.setup(state)
        delta_lambda = tf.nn.softplus(delta_lambda)
        return delta_lambda

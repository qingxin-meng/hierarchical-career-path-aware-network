#!/usr/bin/env python3

"""
@author: Qingxin Meng
@since: 2018-08-16
"""

import photinia as ph
from module.Layer0814 import AttentionLayer
from module.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model(ph.Model):
    """模型定义
    """

    def __init__(self,
                 name,
                 company_count,
                 emb_size=150,
                 latent_dim=300,
                 sample_size=50):
        """模型初始化

        :param name: 模型名
        :param company_count: 公司总数
        :param emb_size: 公司embedding维度
        :param state_size: LSTM单元隐藏单元维度
        """
        self._company_count = company_count
        self._emb_size = emb_size
        self._latent_dim = latent_dim
        self._sample_size = sample_size
        ph.Model.__init__(self, name)

    def _build(self):
        # 网络模块定义 --- build
        # self.graph=tf.Graph()
        # with self.graph.as_default():
        self._emb = ph.Linear('EMB', self._company_count, self._emb_size)
        self._LSTM_encode = ph.LSTMCell('LSTM_encode', self._emb_size, self._latent_dim)
        self._attn = AttentionLayer('AttentionLayer', self._latent_dim)
        self._LSTM_decode = ph.LSTMCell('LSTM_decode', self._latent_dim, self._latent_dim)
        self._dense = ph.Linear('Dense', self._latent_dim, self._company_count,
                                w_init=ph.init.RandomUniform(minval=0, maxval=1),
                                b_init=ph.init.RandomUniform(minval=0, maxval=1))

        trainable_variable_list = self._emb.get_trainable_variables() + \
                                  self._LSTM_encode.get_trainable_variables() + \
                                  self._attn.get_trainable_variables() + \
                                  self._LSTM_decode.get_trainable_variables() + \
                                  self._dense.get_trainable_variables()

        self.l2_norm = tf.global_norm(trainable_variable_list)
        # 输入定义

        self.seq_time = tf.placeholder(
            shape=(None, None,),
            dtype=tf.float32
        )
        self.seq_time_interval = tf.placeholder(
            shape=(None, None, None,),
            dtype=tf.float32
        )

        self.seq_event = tf.placeholder(
            shape=(None, None,),
            dtype=tf.int32
        )
        self.seq_mask = tf.placeholder(
            shape=(None, None,),
            dtype=tf.float32
        )

        self.sample_time_interval = tf.placeholder(
            shape=(None, None, None,),
            dtype=tf.float32
        )
        self.sample_index = tf.placeholder(
            shape=(None, None,),
            dtype=tf.int32
        )
        self.sample_mask = tf.placeholder(
            shape=(None, None,),
            dtype=tf.float32
        )

        # self.seq_mask=tf.ones_like(self.seq_mask,dtype=tf.float32)
        # self.sample_mask=tf.ones_like(self.sample_mask,dtype=tf.float32)

        self.batch_size = tf.shape(self.seq_event)[0]
        self.seq_length = tf.shape(self.seq_event)[1]
        self.seq_event_onehot = tf.one_hot(self.seq_event, depth=self._company_count)



        seq_emb_event = tf.map_fn(
            fn=self.emb_process,
            elems=self.seq_event_onehot
        )

        self.encode_states, self.encode_cell_states = self._LSTM_encode.setup_sequence(seq_emb_event,
                                                                                       return_cell_states=True)
        self.states = tf.map_fn(
            fn=self.relu_process,
            elems=self.encode_states
        )
        tf.summary.histogram('encode_states', self.states)

        # self.states = states * self.seq_mask[:, :, None]

        # self.states=tf_print(self.states,[self.states],'states')

        self.time_diffs = tf.random_poisson(lam=0.01, shape=(self.batch_size, self._sample_size))

        self.compute_loss()

        self.compute_predict_loss()

        train = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

        merged = tf.summary.merge_all()

        self._add_slot(
            'train',
            outputs=(self.loss, merged),
            inputs=(self.seq_time, self.seq_time_interval, self.seq_event, self.seq_mask, self.sample_time_interval,
                    self.sample_index, self.sample_mask),
            updates=train
        )

        self._add_slot(
            'predict_log',
            outputs=(self.log_likelihood_seq, self.log_likelihood_type, self.log_likelihood_time, merged),
            inputs=(self.seq_time, self.seq_time_interval, self.seq_event, self.seq_mask, self.sample_time_interval,
                    self.sample_index, self.sample_mask),
        )

        self._add_slot(
            'predict_loss',
            outputs=(self.log_likelihood_type_predict, self.num_of_errors, self.time_square_errors, merged),
            inputs=(self.seq_time, self.seq_time_interval, self.seq_event, self.seq_mask, self.sample_time_interval,
                    self.sample_index, self.sample_mask),
        )

    def emb_process(self, event):
        emb = self._emb.setup(event)
        return emb

    def relu_process(self, states):
        re = tf.nn.relu(states)
        return re

    def density_process(self, state):
        prob = self._dense.setup(state)
        return prob

    def softplus_scale_process(self, state):
        state = tf.nn.relu(state)
        prob = state / tf.reduce_sum(state, -1, keepdims=True)
        return prob

    def compute_loss(self, ):
        time_since_start_to_end = tf.cumsum(self.seq_time, -1)[:, -1]
        lambda_over_seq = self.compute_lambda_over_seq(self.seq_time_interval)
        lambda_over_seq *= self.seq_mask[:, 1:, None]
        # lambda_over_seq=tf_print(lambda_over_seq,[lambda_over_seq],'lambda_over_seq')
        term1 = tf.reduce_sum(tf.log(1e-9 + tf.reduce_sum(self.seq_event_onehot[:, 1:, :] * lambda_over_seq, -1)), -1)
        lambda_sum_over_seq = tf.reduce_sum(tf.log(1e-9 + tf.reduce_sum(lambda_over_seq, -1)), -1)
        lambda_sum_over_seq *= self.seq_mask[:, 1:, None]
        lambda_over_sample = self.compute_lambda_over_sample(self.sample_time_interval, self.sample_index)
        lambda_over_sample *= self.sample_mask[:, :, None]
        num_sample = tf.reduce_sum(self.sample_mask, -1)
        term2 = tf.where(tf.equal(num_sample, 0), tf.zeros_like(num_sample),
                         tf.reduce_sum(tf.reduce_sum(lambda_over_sample, -1),
                                       -1) * time_since_start_to_end / num_sample)
        # term2 = tf.reduce_sum(tf.reduce_sum(lambda_over_sample, -1), -1) * time_since_start_to_end / num_sample
        self.log_likelihood_type = tf.reduce_mean(term1 - lambda_sum_over_seq)
        self.log_likelihood_seq = tf.reduce_mean(term1 - term2)
        self.log_likelihood_time = self.log_likelihood_seq - self.log_likelihood_type
        tf.summary.scalar('log_likelihood_seq', self.log_likelihood_seq)
        tf.summary.scalar('log_likelihood_type', self.log_likelihood_type)
        tf.summary.scalar('log_likelihood_time', self.log_likelihood_time)
        # self.loss = -self.log_likelihood_seq
        self.loss = -self.log_likelihood_seq + self.l2_norm

    def compute_lambda_over_seq(self, seq_time_interval):
        index = tf.tile(tf.range(1, self.seq_length, 1, dtype=tf.int32)[None, :], [self.batch_size, 1])
        attn_over_seq = self._attn.setup_sequence(seq_time_interval, index, self.seq_mask, self.states)
        # tf.summary.histogram('attn_over_seq', attn_over_seq)
        decode_state = self._LSTM_decode.setup_sequence(attn_over_seq,
                                                        init_cell_state=self.encode_cell_states[:, -1, :],
                                                        init_state=self.encode_states[:, -1, :])
        # tf.summary.histogram('decode_state', decode_state)

        dense_over_seq = tf.map_fn(
            fn=self.density_process,
            elems=decode_state
        )

        lambda_over_seq = tf.map_fn(
            fn=self.softplus_scale_process,
            elems=dense_over_seq
        )

        # tf.summary.histogram('dense_over_seq', dense_over_seq)
        # tf.summary.histogram('lambda_over_seq', lambda_over_seq)
        return lambda_over_seq

    def compute_lambda_over_sample(self, sample_time_interval, sample_index):
        attn_over_sample = self._attn.setup_sequence(sample_time_interval, sample_index, self.seq_mask, self.states)
        # tf.summary.histogram('attn_over_sample', attn_over_sample)
        state_size = tf.shape(self.encode_cell_states)[2]
        batch_size = tf.shape(self.encode_cell_states)[0]
        seq_length = tf.shape(self.seq_event)[1]
        sample_length = tf.shape(sample_time_interval)[1]
        temp_cell = tf.reshape(self.encode_cell_states, [-1, state_size])
        temp_state = tf.reshape(self.encode_states, [-1, state_size])
        multi = tf.tile(tf.range(0, batch_size, 1, dtype=tf.int32)[:, None], [1, sample_length])
        temp_index = multi * seq_length + sample_index - 1
        pre_cells = tf.transpose(tf.gather(temp_cell, temp_index), (1, 0, 2))
        pre_states = tf.transpose(tf.gather(temp_state, temp_index), (1, 0, 2))
        _, decode_state = tf.map_fn(
            fn=lambda x: self._LSTM_decode.setup(x[0], x[1], x[2]),
            elems=(attn_over_sample, pre_cells, pre_states),
            dtype=(tf.float32, tf.float32)
        )

        dense_over_sample = tf.map_fn(
            fn=self.density_process,
            elems=decode_state
        )
        lambda_over_sample = tf.map_fn(
            fn=self.softplus_scale_process,
            elems=dense_over_sample
        )

        lambda_over_sample = tf.transpose(lambda_over_sample, (1, 0, 2))
        # tf.summary.histogram('dense_over_sample', dense_over_sample)
        # tf.summary.histogram('lambda_over_sample', lambda_over_sample)
        return lambda_over_sample

    def predict_each_step(self, seq_index):
        time_cum = tf.cumsum(self.time_diffs, -1)[:, :, None]  # [batch,sample_size]
        temp = tf.cumsum(self.seq_time[:, :seq_index], -1, reverse=True)[:, None, :]  # [batch,seq_observe]
        time_span_on_sample = time_cum + temp  # [batch,sample_size,seq_observe]
        seq_index_on_sample = tf.ones([self.batch_size, self._sample_size], dtype=tf.int32) * seq_index
        lambda_over_sample_sequence = self.compute_lambda_over_sample(time_span_on_sample, seq_index_on_sample)
        # lambda_over_sample_sequence *= self.seq_mask[:,seq_index][:,None,None]
        lambda_sum_over_sample_sequence = tf.reduce_sum(lambda_over_sample_sequence, -1)  # [batch,sample_size]
        term_1 = self.time_diffs
        cum_num = tf.range(start=1, limit=self._sample_size + 1, delta=1, dtype=tf.float32)[None, :]
        # [batch,sample_size]
        term_2 = tf.exp(
            (
                    -1.0 * tf.cumsum(
                lambda_sum_over_sample_sequence, axis=1
            ) / cum_num
            ) * self.time_diffs
        )
        # size_batch * sample_size
        density = term_2 * lambda_sum_over_sample_sequence
        # size_batch * sample_size

        time_prediction_since_last_event = tf.reduce_mean(
            term_1 * density, axis=1
        ) * self.time_diffs[:, -1]

        # predict next event type without knowledge of happened times
        lambda_each_step_over_sims = lambda_over_sample_sequence * density[:, :,
                                                                   None] / lambda_sum_over_sample_sequence[:, :, None]

        # size_batch * sample_size*company_count
        prob_over_type = tf.reduce_mean(
            lambda_each_step_over_sims, axis=1
        ) * tf.cumsum(self.time_diffs, -1)[:, -1][:, None]



        # size_batch * dim_process
        return prob_over_type, time_prediction_since_last_event

    def compute_predict_loss(self):
        seq_index = tf.range(1, self.seq_length, 1)
        prob_over_type_over_seq, time_predict_over_seq = tf.map_fn(
            fn=lambda x: self.predict_each_step(x),
            elems=seq_index,
            dtype=(tf.float32, tf.float32)
        )

        target_type_onehot = self.seq_event_onehot[:, 1:, :]
        target_time = self.seq_time[:, 1:]
        target_type = self.seq_event[:, 1:]
        # Type first
        prob_over_seq = tf.reduce_sum(tf.transpose(prob_over_type_over_seq, (1, 0, 2)) * target_type_onehot, 2)
        log_prob_over_seq = tf.log(
            prob_over_seq + 1e-9
        )
        log_prob_over_seq *= self.seq_mask[:, 1:]
        self.log_likelihood_type_predict = tf.reduce_mean(tf.reduce_sum(log_prob_over_seq, -1))  # [batch,]
        #
        # Time
        diff_time = (target_time - tf.transpose(time_predict_over_seq, (1, 0))) ** 2
        diff_time *= self.seq_mask[:, 1:]

        self.num_of_events = tf.reduce_sum(self.seq_mask[:, 1:], -1)
        # TODO: Hamming loss for prediction checking
        #
        type_prediction = tf.transpose(tf.argmax(
            prob_over_type_over_seq, axis=2, output_type=tf.int32), (1, 0))  # [batch,seq]

        diff_type = tf.cast(tf.abs(
            target_type - type_prediction
        ), tf.float32) * self.seq_mask[:, 1:]

        diff_type = tf.where(
            tf.greater_equal(diff_type, 0.5),
            tf.ones_like(diff_type), tf.zeros_like(diff_type)
        )
        self.num_of_errors = tf.reduce_mean(tf.reduce_sum(diff_type, -1) / self.num_of_events)
        self.time_square_errors = tf.reduce_mean(tf.reduce_sum(diff_time, -1) / self.num_of_events)

        #
        # self.predict_loss = -tf.reduce_mean(self.log_likelihood_type_predict / self.num_of_events) + tf.reduce_mean(
        #     self.square_errors / self.num_of_events)
        # self.cost_to_optimize = -self.log_likelihood_type_predict + self.term_reg

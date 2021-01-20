import tensorflow as tf
import os
import numpy as np
import sys
import pickle as cPickle
import json
from ipdb import set_trace

def tf_print(op, tensors, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
        return x

    prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op

def deserialize_from_file(path):
    f = open(path, 'rb')
    obj = cPickle.load(f)
    f.close()
    return obj

def serialize_to_file(obj, path, protocol=2):
    f = open(path, 'wb')
    cPickle.dump(obj, f, protocol=protocol)
    f.close()


def export_result_file(result_path,file_name,obj):
    file= result_path+file_name+'.pkl'
    with open(file,'wb') as f:
        cPickle.dump(obj,f)
    print('file export done!')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def flatten(tensor):
    shape = tensor.get_shape().as_list()  # a list: [None, 9, 2]
    dim = np.prod(shape[1:])  # dim = prod(9,2) = 18
    new_tensor = tf.reshape(tensor, [-1, dim])
    return new_tensor

def predict_accuracy(delta_lambda,target_time,segmentsize,mask):
    """
    :param predict_type15: [batch,seq-1,split_k]
    :param delta_lambda:  [batch,seq-1,segment_size]
    :param predict_num:
    :param seq_event: [batch,seq]
    :param seq_time: [batch,seq]
    :param seq_mask: [batch,seq]
    :return: type accuracy,time_hamming,time_rmse
    """
    target_time=(target_time+0.5)*mask
    predict_time=compute_survival_time(delta_lambda,segmentsize)*mask
    time_hamming=np.sum(np.abs(predict_time-target_time))
    time_pow=np.sum((predict_time-target_time)**2)
    return time_hamming,time_pow

def compute_survival_time(delta_lambda,segmentsize):
    cum_delta_lambda = np.cumsum(delta_lambda, -1)
    cum_index = np.arange(0.5,segmentsize+0.5,1)[None, None, :]
    predict_time = np.sum(cum_index * delta_lambda * np.exp(-cum_delta_lambda), -1)
    return predict_time

def viterbi_decode(score, transition_params,top_k):
  """Decode the highest scoring sequence of tags outside of TensorFlow.
  This should only be used at test time.
  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()
  viterbi_score = np.max(trellis[-1])

  viterbi_topk=[]
  for t in range(score.shape[0]):
      temp=np.argsort(trellis[t])[-top_k:]
      viterbi_topk.append(temp)
  return viterbi, viterbi_score,viterbi_topk

def compute_viterbi_score(seq_event,score,transition_params):
    trellis = np.zeros_like(score)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + v[seq_event[t]]

    return trellis


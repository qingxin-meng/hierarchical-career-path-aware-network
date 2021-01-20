import photinia as ph
import time
import pickle
import random
from module.Data1207 import Data
from module.utils import *
from module.LSTM_with_inout_matrix import Career
from module.Evaluate import evaluate

# from tensorflow.python import debug as tf_debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


max_loop = 50
company_count = 1002
max_time = 21
emb_size = 100
emb_time_size = 10
emb_in_size=50
emb_out_size=10
latent_dim = 200
batch_size = 32
topk = 15
keep_prob = 0.8

source_path = os.path.dirname(os.getcwd()) + '/source_data/seq_data1002.pkl'
model_save_path = os.path.dirname(os.getcwd()) + '/model_saved/LSTM_with_inout_matrix_.ckpt'
result_path = os.path.dirname(os.getcwd()) + '/result/Career_.txt'
data_saved_path = os.path.dirname(os.getcwd()) + '/data_saved/data1002.pkl'

tag1 = 'train'
tag2 = 'valid'
tag3 = 'test'
model = Career('Career',
              company_count=company_count,
              max_time=max_time,
              emb_size=emb_size,
              emb_time_size=emb_time_size,
              emb_in_size=emb_in_size,
              emb_out_size=emb_out_size,
              latent_dim=latent_dim,
              keep_prob=keep_prob)

train = model.get_slot('train')
predict = model.get_slot('predict')
print('the model is loaded!')



# data = Data(source_path, company_count, max_time, batch_size=batch_size, train_proportion=0.8,
#                 valid_proportion=0.1, test_proportion=0.1,seed=49)
# serialize_to_file(data, data_saved_path)
data = deserialize_from_file(data_saved_path)

print('data is prepared!')

saver = tf.train.Saver(max_to_keep=50)

print('train beginning...')
with ph.get_session() as sess:
    start = time.time()
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, os.path.dirname(os.getcwd()) + "/model_saved/./LSTM_with_inout_matrix.ckpt-0")
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    for n_loop in range(max_loop):
        total_batch = len(data.batch_ind[tag1])
        data_gen = data.gen_batch(tag1)
        for n in range(total_batch):
            input=next(data_gen)
            loss ,loglike_event,loglike_duration,delta_lambda, prob_next_event = \
                model.train(input[0],input[1],input[2],input[3],input[4],input[5],input[6],input[7])
            if n % 100 == 0:
                print('the train loss:{},loglike_event:{},loglike_time:{}'.format(loss,loglike_event,loglike_duration))
                end = time.time()
                train_duration = (end - start) / 60
                print('train the {}th loop {}th batch need cum time {} minutes'.format(n_loop, n, train_duration))
        saver.save(sess, model_save_path, global_step=n_loop)
        print('model saved!')

        print('validation beginning!')
        total_valid_batch=len(data.batch_ind[tag2])
        total_test_batch=len(data.batch_ind[tag3])
        valid_data_gen=data.gen_batch(tag2)
        test_data_gen=data.gen_batch(tag3)
        valid_eval=evaluate(topk,max_time)
        test_eval=evaluate(topk,max_time)

        for n in range(total_valid_batch):
            print('epoch:{},valid batch:{}'.format(n_loop,n))
            input=next(valid_data_gen)
            delta_lambda,  prob_next_event = \
                model.predict(input[0],input[1],input[2],input[3],
                              input[4],input[5],input[6],input[7])
            valid_eval.batch_evaluate(delta_lambda,prob_next_event,input[2],input[3],input[4])

            print('epoch:{},test batch:{}'.format(n_loop, n))
            input = next(test_data_gen)
            delta_lambda,  prob_next_event = \
                model.predict(input[0], input[1], input[2], input[3],
                              input[4], input[5], input[6], input[7])
            test_eval.batch_evaluate(delta_lambda, prob_next_event, input[2], input[3], input[4])

        pre_text='valid loop:%d '%n_loop
        valid_eval.epoch_evaluate(output_file_path=result_path,pre_text=pre_text)
        pre_text = 'test loop:%d ' % n_loop
        test_eval.epoch_evaluate(output_file_path=result_path, pre_text=pre_text)
        # with open(os.path.dirname(os.getcwd()) + '/result/valid_eval_loop{}.pkl'.format(n_loop),'wb') as f:
        #     pickle.dump(valid_eval,f)
        #
        # with open(os.path.dirname(os.getcwd()) + '/result/test_eval_loop{}.pkl'.format(n_loop),'wb') as f:
        #     pickle.dump(test_eval,f)

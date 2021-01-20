import numpy as np
import random
import pickle
from module.utils import *
from sklearn.preprocessing import normalize


class Data():
    def __init__(self, source_path, company_count, time_segment, train_batch_size=32,valid_batch_size=256, train_proportion=0.8,
                 valid_proportion=0.1, test_proportion=0.1,seed=None):
        if train_proportion + valid_proportion + test_proportion != 1:
            raise Exception("invalid data splitting!", train_proportion, train_proportion, test_proportion)
        data = deserialize_from_file(source_path)
        self.time_segment=time_segment
        self.size={}
        events = data[0]
        times = data[1]
        starts=data[2]
        ends=data[3]
        if seed is None:
            randnum = random.randint(0, 100)
        else:
            randnum=seed
        random.seed(randnum)
        random.shuffle(events)
        random.seed(randnum)
        random.shuffle(times)
        random.seed(randnum)
        random.shuffle(starts)
        random.seed(randnum)
        random.shuffle(ends)
        self.size['total_sample'] = len(events)
        print('the total sample is:{}'.format(self.size['total_sample']))
        self.size['train'] = int(np.ceil(self.size['total_sample'] * train_proportion))
        self.size['valid'] = int(np.ceil(self.size['total_sample'] * valid_proportion))
        self.size['test'] = self.size['total_sample'] - self.size['train'] - self.size['valid']
        self.batch_ind = {}
        self.batch_ind['train'] = np.arange(int(np.ceil(self.size['train'] / train_batch_size)))
        self.batch_ind['valid'] = np.arange(int(np.ceil(self.size['valid'] / valid_batch_size)))
        self.batch_ind['test'] = np.arange(int(np.ceil(self.size['test'] / valid_batch_size)))
        self.split_events={}
        self.split_times={}
        self.split_events['train']=events[:self.size['train']]
        self.split_events['valid']=events[self.size['train']:(self.size['train']+self.size['valid'])]
        self.split_events['test']=events[(self.size['train']+self.size['valid']):]
        self.split_times['train'] = times[:self.size['train']]
        self.split_times['valid'] = times[self.size['train']:(self.size['train'] + self.size['valid'])]
        self.split_times['test'] = times[(self.size['train'] + self.size['valid']):]

        print('the total train batch is:{}'.format(len(self.batch_ind['train'])))
        print('the total valid batch is:{}'.format(len(self.batch_ind['valid'])))
        print('the total test batch is:{}'.format(len(self.batch_ind['test'])))

        self.batches = {'train': {'train_event': {}, 'train_time': {},'start':{},'end':{}, 'mask': {},
                                  'target_event': {}, 'target_time': {}},
                        'valid': {'train_event': {}, 'train_time': {},'start':{},'end':{}, 'mask': {},
                                  'target_event': {}, 'target_time': {}},
                        'test': {'train_event': {}, 'train_time': {},'start':{},'end':{}, 'mask': {},
                                  'target_event': {}, 'target_time': {}}}

        ## dimission statistics
        duration_matrix = np.zeros([company_count, time_segment])
        ## company_in_matrix 的size应该是[company_count,61],多加一维是为了后面截取往后十年的流出率时，超出年限（超过2018年）的部分能填充0对齐
        company_in_matrix = np.zeros([company_count, 62])
        company_out_matrix=np.zeros([company_count,62])
        for one_seq_events, one_seq_times,one_seq_start,one_seq_end\
                in zip(events[:self.size['train']], times[:self.size['train']],starts[:self.size['train']],ends[:self.size['train']]):
            for com, tim,start,end in zip(one_seq_events, one_seq_times,one_seq_start,one_seq_end):
                duration_matrix[com, tim] += 1
                company_in_matrix[com,start] +=1
                company_out_matrix[com,end] +=1
        self.duration_matrix=normalize(duration_matrix,norm='l1',axis=0)
        self.company_in_matrix=normalize(company_in_matrix,norm='l1',axis=0)
        self.company_out_matrix=normalize(company_out_matrix,norm='l1',axis=0)
        print('the data batch is prepared!')

        for tag in ['train','valid','test']:
            if tag=='train':
                offset=0
                cap=self.size['train']
                batch_size=train_batch_size
            elif tag=='valid':
                offset=self.size['train']
                cap=self.size['train']+self.size['valid']
                batch_size=valid_batch_size
            else :
                offset=self.size['train']+self.size['valid']
                cap=self.size['total_sample']
                batch_size=valid_batch_size
            for batch_index in self.batch_ind[tag]:
                i, j = batch_index * batch_size+offset, min((batch_index + 1) * batch_size+offset, cap)
                max_length = max([len(seq) for seq in events[i:j]]) - 1
                self.batches[tag]['train_event'][batch_index] = np.array(
                    [seq[:-1] + (max_length - len(seq[:-1])) * [0] for seq in events[i:j]])
                self.batches[tag]['train_time'][batch_index] = np.array(
                    [seq[:-1] + (max_length - len(seq[:-1])) * [0] for seq in times[i:j]])
                self.batches[tag]['mask'][batch_index] = np.array(
                    [len(seq[:-1]) * [1] + (max_length - len(seq[:-1]))* [0] for seq in events[i:j]])
                self.batches[tag]['target_event'][batch_index] = np.array(
                    [seq[1:] + (max_length - len(seq[1:])) * [0] for seq in events[i:j]])
                self.batches[tag]['target_time'][batch_index] = np.array(
                    [seq[1:] + (max_length - len(seq[1:])) * [0] for seq in times[i:j]])
                self.batches[tag]['start'][batch_index]=np.array(
                    [seq[1:] + (max_length - len(seq[1:])) * [0] for seq in starts[i:j]]
                )
                ## 用下一个公司开始时间的company_in 数据预测下一个跳转公司
                ## 用下一个公司开始时间,及已知跳转的公司，并往后十年的该公司company_out数据预测离职时间

    def gen_batch(self,tag):
        for ind in np.random.permutation(self.batch_ind[tag]):
            seq_event=self.batches[tag]['train_event'][ind]  #[batch,seq]
            seq_time=self.batches[tag]['train_time'][ind]   #[batch,seq]
            seq_mask=self.batches[tag]['mask'][ind]   #[batch,seq]
            target_event=self.batches[tag]['target_event'][ind]
            target_time=self.batches[tag]['target_time'][ind]
            seq_start=self.batches[tag]['start'][ind]
            seq_start_pre=np.where(seq_start-1>=0,seq_start-1,61)
            seq_company_in=self.company_in_matrix[:,seq_start_pre] #[company,batch,seq]
            seq_company_in=np.transpose(seq_company_in,(1,2,0))*seq_mask[:,:,None] #[batch,seq,company]
            temp=[]
            for t in [-2,-1]:
                temp_time=np.where(seq_start+t>=0,seq_start+t,61)
                temp_out=self.company_out_matrix[seq_event,temp_time]
                temp.append(temp_out)
            seq_company_out=np.stack(temp,0)
            seq_company_out=np.transpose(seq_company_out,(1,2,0))*seq_mask[:,:,None]  #[batch,seq,2]
            seq_duration_prob=self.duration_matrix[seq_event,:]*seq_mask[:,:,None]  #[batch,seq,2]

            yield seq_event,seq_time,seq_mask,target_event,target_time,seq_company_in,seq_company_out,seq_duration_prob

import numpy as np
import random
import pickle
from module.utils import *
from ipdb import set_trace

class Data():
    def __init__(self, company_count=1002, time_segment=21, train_batch_size=64,valid_batch_size=64,valid_proportion=0.1, split_year=2010,seed=None):
        with open('../source_data/linkedin6.pkl','rb') as f:
            self.source_seq=pickle.load(f)
        with open('../source_data/company_info_dict.pkl','rb') as f:
            self.company_info_dict=pickle.load(f)
        with open('../source_data/in_degree_agg3year.pkl','rb') as f:
            self.in_degree=cPickle.load(f,encoding='bytes')
        with open('../source_data/out_degree_agg3year.pkl','rb') as f:
            self.out_degree=cPickle.load(f,encoding='bytes')
        with open('../source_data/hop_degree_agg3year.pkl','rb') as f:
            self.hop_degree=cPickle.load(f,encoding='bytes')

        self.person_ids = list(self.source_seq.keys())
        if seed is None:
            seed = random.randint(0, 100)
        random.seed(seed)
        random.shuffle(self.person_ids)

        self.train_dict_after_mask={}
        self.size = {}
        self.select_ids = {'train': [], 'valid': [], 'test': []}
        self.split_year_id = (split_year - 1991) * 2
        for p_id in self.person_ids:
            seq_year_mask = self.get_year_mask(p_id)
            if seq_year_mask.sum()>0:
                seq_company_length = seq_year_mask.sum()
                seq_comany = np.array(self.source_seq[p_id]['company'])[seq_year_mask]
                seq_company_dur = np.array(self.source_seq[p_id]['company_dur'])[seq_year_mask]
                seq_start = np.array(self.source_seq[p_id]['start'])[seq_year_mask]
                seq_idx = np.array(self.source_seq[p_id]['idx'])[seq_year_mask]
                seq_pos_length = seq_idx[-1]+1
                seq_working_month = np.array(self.source_seq[p_id]['working_months'])[seq_year_mask]
                seq_pos = np.array(self.source_seq[p_id]['pos'])[:seq_pos_length]
                seq_pos_dur = np.array(self.source_seq[p_id]['pos_dur'])[:seq_pos_length]
                if p_id not in self.train_dict_after_mask:
                    self.train_dict_after_mask[p_id]={'year_mask':seq_year_mask,'company_length':seq_company_length,
                                                      'company':seq_comany,'company_dur':seq_company_dur,
                                                      'start':seq_start,'pos_idx':seq_idx,'pos_length':seq_pos_length,
                                                      'working_month':seq_working_month,'pos':seq_pos,
                                                      'pos_dur':seq_pos_dur}
        self.select_ids['train']=list(self.train_dict_after_mask.keys())
        self.size['total_sample'] = len(self.select_ids['train'])
        self.size['train'] = self.size['total_sample']
        self.size['valid'] = int(np.ceil(self.size['total_sample'] * valid_proportion))
        self.size['test'] = self.size['valid']

        m=0
        for p_id in self.person_ids:
            m+=1
            if self.get_target_mask(p_id).sum() > 0:
                self.select_ids['valid'].append(p_id)
            if len(self.select_ids['valid'])>=self.size['valid']:
                break

        for p_id in self.person_ids[m:]:
            if self.get_target_mask(p_id).sum() > 0:
                self.select_ids['test'].append(p_id)
            if len(self.select_ids['test']) >= self.size['test']:
                break

        self.time_segment=time_segment
        self.train_batch_size=train_batch_size
        self.valid_batch_size=valid_batch_size
        self.batch_ind = {}


        self.batch_ind['train'] = np.arange(int(np.ceil(self.size['train'] / train_batch_size)))
        self.batch_ind['valid'] = np.arange(int(np.ceil(self.size['valid'] / valid_batch_size)))
        self.batch_ind['test'] = np.arange(int(np.ceil(self.size['test'] / valid_batch_size)))

        print('the total sample is:{}'.format(len(self.person_ids)))
        print('the total train batch is:{}'.format(len(self.batch_ind['train'])))
        print('the total valid batch is:{}'.format(len(self.batch_ind['valid'])))
        print('the total test batch is:{}'.format(len(self.batch_ind['test'])))



        print('the data batch is loaded!')

    def get_year_mask(self,idx):
        start = self.source_seq[idx]['start']
        year_mask = np.array([y < self.split_year_id for y in start])
        return year_mask

    def get_pading_length(self,max_length,p_id,tag='company'):
        if tag=='company':
            re=max_length - self.train_dict_after_mask[p_id]['company_length']
            return re
        elif tag=='pos':
            re=max_length-self.train_dict_after_mask[p_id]['pos_length']
            return re
        else:
            raise Exception('invalid tag value!')


    def padding_zero_to_array(self,array,padding_length):
        if padding_length > 0:
            padding_zero=np.zeros(shape=[padding_length])
            re=np.concatenate([array,padding_zero],-1)
            return re
        else:
            return array

    def get_target_mask(self,idx):
        start = self.source_seq[idx]['start']
        target_mask = np.array([y >= self.split_year_id for y in start])
        target_mask=np.where(target_mask == True, 1, 0)[1:]
        return target_mask


    def gen_batch(self,tag):
        if tag not in ['train','valid','test']:
            raise Exception('invalid tag value!')
        if tag =='train':
            offset=0
            batch_size=self.train_batch_size
            cap=self.size['train']
        elif tag =='valid':
            offset = 0
            batch_size=self.valid_batch_size
            cap= self.size['valid']
        else:
            offset = self.size['valid']
            batch_size=self.valid_batch_size
            cap=self.size['test']+self.size['valid']
        # initialize

        if tag=='train':
            for ind in self.batch_ind[tag]:
                batch_person_connect = []
                batch_person_des = []
                batch_com_des = []
                batch_com_size = []
                batch_com_finacial = []
                batch_com_age = []
                batch_com_location = []

                i,j=ind*batch_size+offset,min((ind+1)*batch_size+offset,cap)

                max_com_length=max(self.train_dict_after_mask[p_id]['company_length'] for p_id in self.select_ids['train'][i:j])
                max_pos_length=max(self.train_dict_after_mask[p_id]['pos_length'] for p_id in self.select_ids['train'][i:j])

                batch_com = np.array([self.padding_zero_to_array (self.train_dict_after_mask[p_id]['company'],self.get_pading_length(max_com_length,p_id))
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)

                batch_com_dur = np.array([self.padding_zero_to_array (self.train_dict_after_mask[p_id]['company_dur'],self.get_pading_length(max_com_length,p_id))
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)

                batch_start = np.array([self.padding_zero_to_array (self.train_dict_after_mask[p_id]['start'],self.get_pading_length(max_com_length,p_id))
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)

                batch_idx = np.array([self.padding_zero_to_array (self.train_dict_after_mask[p_id]['pos_idx'],self.get_pading_length(max_com_length,p_id))
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)

                batch_com_mask = np.array([(self.train_dict_after_mask[p_id]['company_length']-1)*[1] + self.get_pading_length(max_com_length,p_id) * [0]
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)

                batch_working_month=np.array([self.padding_zero_to_array (self.train_dict_after_mask[p_id]['working_month'],self.get_pading_length(max_com_length,p_id))
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)

                batch_pos=np.array([self.padding_zero_to_array (self.train_dict_after_mask[p_id]['pos'],self.get_pading_length(max_pos_length,p_id,'pos'))
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)
                batch_pos_dur=np.array([self.padding_zero_to_array (self.train_dict_after_mask[p_id]['pos_dur'],self.get_pading_length(max_pos_length,p_id,'pos'))
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)
                batch_pos_mask=np.array([self.train_dict_after_mask[p_id]['pos_length']*[1] + self.get_pading_length(max_pos_length,p_id,'pos') * [0]
                                      for p_id in self.select_ids['train'][i:j]]).astype(int)

                batch_target_mask=np.ones_like(batch_com_mask)

                for p_id in self.select_ids['train'][i:j]:
                    if self.source_seq[p_id]['connection'] is not None:
                        batch_person_connect.append(self.source_seq[p_id]['connection'])  # [batch]
                        batch_person_des.append(self.source_seq[p_id]['des'])  # [batch,50]
                    else:
                        batch_person_connect.append(0)
                        batch_person_des.append(np.zeros([50]))

                    seq_com_des = []
                    seq_com_size = []
                    seq_com_finacial = []
                    seq_com_age = []
                    seq_com_location = []
                    for com in self.train_dict_after_mask[p_id]['company']:
                        if com in self.company_info_dict:
                            des  = self.company_info_dict[com]['des']
                            size = self.company_info_dict[com]['size']
                            finacial= self.company_info_dict[com]['finacial']
                            age= self.company_info_dict[com]['age']
                            location= self.company_info_dict[com]['location']

                            seq_com_des.append(des)
                            seq_com_size.append(size)
                            seq_com_finacial.append(finacial)
                            seq_com_age.append(age)
                            seq_com_location.append(location)
                        else:
                            seq_com_des.append(np.zeros(50))
                            seq_com_size.append(0)
                            seq_com_finacial.append(0)
                            seq_com_age.append(0)
                            seq_com_location.append(0)

                    batch_com_des.append(seq_com_des)
                    batch_com_size.append(seq_com_size)
                    batch_com_finacial.append(seq_com_finacial)
                    batch_com_age.append(seq_com_age)
                    batch_com_location.append(seq_com_location)

                batch_person_connect=np.array(batch_person_connect)
                batch_person_des=np.stack(batch_person_des,0)

                batch_com_size = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_size])
                batch_com_finacial = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_finacial])
                batch_com_age = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_age])
                batch_com_location = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_location])
                batch_com_des = np.stack([seq_array+(max_com_length - len(seq_array)) * [[0] * 50] for seq_array in batch_com_des],axis=0)

                batch_in=self.in_degree[batch_start,:] #[batch,seq,company]
                batch_out=self.out_degree[batch_start,:]
                batch_trans=self.hop_degree[batch_start,batch_com,:]


                yield batch_com,batch_com_dur,batch_com_mask,\
                      batch_idx,batch_working_month,\
                      batch_pos,batch_pos_dur,batch_pos_mask,\
                      batch_com_size,batch_com_finacial,batch_com_age,batch_com_location,batch_com_des,\
                      batch_in,batch_out,batch_trans,batch_person_connect,batch_person_des,batch_target_mask

        else:
            for ind in self.batch_ind[tag]:
                batch_person_connect = []
                batch_person_des = []
                batch_com_des = []
                batch_com_size = []
                batch_com_finacial = []
                batch_com_age = []
                batch_com_location = []

                i, j = ind * batch_size + offset, min((ind + 1) * batch_size + offset, cap)

                max_com_length = max(self.source_seq[p_id]['company_length'] for p_id in self.select_ids[tag][i:j])
                max_pos_length = max(self.source_seq[p_id]['pos_length'] for p_id in self.select_ids[tag][i:j])

                batch_com = np.array(
                    [self.source_seq[p_id]['company'] + (max_com_length - self.source_seq[p_id]['company_length']) * [0]
                     for p_id in self.select_ids[tag][i:j]])
                batch_com_dur = np.array([self.source_seq[p_id]['company_dur'] + (
                            max_com_length - self.source_seq[p_id]['company_length']) * [0]
                                          for p_id in self.select_ids[tag][i:j]])
                batch_start = np.array(
                    [self.source_seq[p_id]['start'] + (max_com_length - self.source_seq[p_id]['company_length']) * [0]
                     for p_id in self.select_ids[tag][i:j]])
                batch_idx = np.array(
                    [self.source_seq[p_id]['idx'] + (max_com_length - self.source_seq[p_id]['company_length']) * [0]
                     for p_id in self.select_ids[tag][i:j]])
                batch_com_mask = np.array([(self.source_seq[p_id]['company_length'] - 1) * [1] + (
                            max_com_length - self.source_seq[p_id]['company_length']) * [0]
                                           for p_id in self.select_ids[tag][i:j]])
                batch_working_month = np.array([self.source_seq[p_id]['working_months'] + (
                            max_com_length - self.source_seq[p_id]['company_length']) * [0]
                                                for p_id in self.select_ids[tag][i:j]])

                batch_target_mask=np.array([self.padding_zero_to_array(self.get_target_mask(p_id),max_com_length - self.source_seq[p_id]['company_length'])
                                                for p_id in self.select_ids[tag][i:j]])

                batch_pos = np.array(
                    [self.source_seq[p_id]['pos'] + (max_pos_length - self.source_seq[p_id]['pos_length']) * [0]
                     for p_id in self.select_ids[tag][i:j]])
                batch_pos_dur = np.array(
                    [self.source_seq[p_id]['pos_dur'] + (max_pos_length - self.source_seq[p_id]['pos_length']) * [0]
                     for p_id in self.select_ids[tag][i:j]])
                batch_pos_mask = np.array([self.source_seq[p_id]['pos_length'] * [1] + (
                            max_pos_length - self.source_seq[p_id]['pos_length']) * [0]
                                           for p_id in self.select_ids[tag][i:j]])

                for p_id in self.select_ids[tag][i:j]:
                    if self.source_seq[p_id]['connection'] is not None:
                        batch_person_connect.append(self.source_seq[p_id]['connection'])  # [batch]
                        batch_person_des.append(self.source_seq[p_id]['des'])  # [batch,50]
                    else:
                        batch_person_connect.append(0)
                        batch_person_des.append(np.zeros([50]))

                    seq_com_des = []
                    seq_com_size = []
                    seq_com_finacial = []
                    seq_com_age = []
                    seq_com_location = []
                    for com in self.source_seq[p_id]['company']:
                        if com in self.company_info_dict:
                            des = self.company_info_dict[com]['des']
                            size = self.company_info_dict[com]['size']
                            finacial = self.company_info_dict[com]['finacial']
                            age = self.company_info_dict[com]['age']
                            location = self.company_info_dict[com]['location']

                            seq_com_des.append(des)
                            seq_com_size.append(size)
                            seq_com_finacial.append(finacial)
                            seq_com_age.append(age)
                            seq_com_location.append(location)
                        else:
                            seq_com_des.append(np.zeros(50))
                            seq_com_size.append(0)
                            seq_com_finacial.append(0)
                            seq_com_age.append(0)
                            seq_com_location.append(0)

                    batch_com_des.append(seq_com_des)
                    batch_com_size.append(seq_com_size)
                    batch_com_finacial.append(seq_com_finacial)
                    batch_com_age.append(seq_com_age)
                    batch_com_location.append(seq_com_location)

                batch_person_connect = np.array(batch_person_connect)
                batch_person_des = np.stack(batch_person_des, 0)

                batch_com_size = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_size])
                batch_com_finacial = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_finacial])
                batch_com_age = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_age])
                batch_com_location = np.array([seq + (max_com_length - len(seq)) * [0] for seq in batch_com_location])
                batch_com_des = np.stack(
                    [seq_array + (max_com_length - len(seq_array)) * [[0] * 50] for seq_array in batch_com_des], axis=0)

                batch_in = self.in_degree[batch_start, :]  # [batch,seq,company]
                batch_out = self.out_degree[batch_start, :]
                batch_trans = self.hop_degree[batch_start, batch_com, :]

                yield batch_com, batch_com_dur, batch_com_mask, \
                      batch_idx, batch_working_month, \
                      batch_pos, batch_pos_dur, batch_pos_mask, \
                      batch_com_size, batch_com_finacial, batch_com_age, batch_com_location, batch_com_des, \
                      batch_in, batch_out, batch_trans, batch_person_connect, batch_person_des,batch_target_mask
import os
import json
import pickle
import time
import numpy as np
from ipdb import set_trace

def compute_time_span(start_str1,end_str2):
    y0,m0=start_str1.split('/')
    y1,m1=end_str2.split('/')
    re=(int(y1)-int(y0))*12+(int(m1)-int(m0))
    return re

def norm_duration(duration):
    if duration <0:
        return int(0)
    elif duration > 120:
        re = int(20)
        return re
    else:
        re = int(np.floor(duration / 6))
        return re

def time2id(datestr):
    if datestr<'1991/1' or datestr>'2018/12':
        return 56 #56是padding 0的数据
    else:
        y,m=datestr.split('/')
        y,m=int(y),int(m)
        ind=(y-1991)*2+m//6
        return ind

def norm_working_month(month):
    if month<=12:
        return 0
    elif 12<month<=36:
        return 1
    elif 36<month<=60:
        return 2
    elif 60<month<=120:
        return 3
    elif 120<month<=240:
        return 4
    elif 240<month<=360:
        return 5
    else:
        return 6

with open('/Users/qingxinmeng/PycharmProjects/time_sequence_prediction/linkedin5.json','rb') as f:
    source_seq=json.load(f)
with open('/Users/qingxinmeng/PycharmProjects/time_sequence_prediction/source_data/person_info_dict.pkl','rb') as f:
    person_info_dict=pickle.load(f)
total_sample=len(source_seq)
print('there are total {} sample'.format(total_sample))
n=0
person_career_dict={}
for key,content in source_seq.items():
    n +=1
    person_career_dict[key] = {}
    if key in person_info_dict:
        person_career_dict[key]['connection']=person_info_dict[key]['connection']
        person_career_dict[key]['des']=person_info_dict[key]['des']
    else:
        person_career_dict[key]['connection'] = None
        person_career_dict[key]['des'] = None
    person_career_dict[key]['pos_length']=len(content['company'])
    person_career_dict[key]['pos']=content['normalized_title']
    person_career_dict[key]['pos_dur']=[norm_duration(dur) for dur in content['duration']]

    seq_company=[]
    seq_id=[]
    for p_i,p_j in zip(range(len(content['company'])-1),range(1,len(content['company']))):
        if content['company'][p_i] != content['company'][p_j]:
            seq_company.append(content['company'][p_i])
            seq_id.append(p_i)
        if p_j==len(content['company'])-1:
            seq_company.append(content['company'][p_j])
            seq_id.append(p_j)
    com_length=len(seq_company)

    seq_company_dur=[]
    seq_company_start=[]
    seq_working_month=[]

    seq_id_plus1=[i+1 for i in seq_id]
    seq_id_plus1_concate0=[0]+seq_id_plus1[:-1]
    for id0,id1 in zip(seq_id_plus1_concate0,seq_id_plus1):
        start=min(content['start'][id0:id1])
        end=max(content['end'][id0:id1])
        dur=compute_time_span(start,end)
        working_month=compute_time_span(content['first_work_date'],start)
        seq_company_start.append(start)
        seq_company_dur.append(dur)
        seq_working_month.append(working_month)



    person_career_dict[key]['company']=seq_company
    person_career_dict[key]['company_dur']=[norm_duration(dur) for dur in seq_company_dur]
    person_career_dict[key]['start']=[time2id(temp_start) for temp_start in seq_company_start]
    person_career_dict[key]['idx']=seq_id
    person_career_dict[key]['working_months']=[norm_working_month(temp_month) for temp_month in seq_working_month]
    person_career_dict[key]['company_length']=com_length




pickle.dump(person_career_dict, open('/Users/qingxinmeng/PycharmProjects/time_sequence_prediction/source_data/linkedin6.pkl', 'wb'))
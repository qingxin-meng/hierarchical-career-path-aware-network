import pickle as cPickle
import json
from module.utils import *

json_path= os.getcwd()+'/linkedin2.json'
data_path=os.getcwd()+'/source_data/seq_data.pkl'
seq_event=[]
seq_time=[]

c2i = cPickle.load(open(os.getcwd()+'/source_data/c2i.pkl', 'rb'))
data=json.load(open(json_path,'r'))
max_time_since_start=[]
max_time_since_last=0.0
seq_length=[]
for record in data:
    one_seq_event = []
    one_seq_time = []
    start_t=record['month'][0]
    for x,y in zip(record['company'],record['month']):
        if x not in c2i:
            continue
        else:
            one_seq_event.append(c2i[x])
            one_seq_time.append(y)
    temp=[x1-x0 for x1,x0 in zip(one_seq_time[1:],one_seq_time[:-1])]
    one_seq_time =[0]
    one_seq_time.extend(temp)
    if len(one_seq_event)<2:
        continue
    if max(one_seq_time)>720 or np.cumsum(one_seq_time)[-1] >720:
        continue
    else:
        seq_event.append(one_seq_event)
        seq_time.append(one_seq_time)
        seq_length.append(len(one_seq_event))
        max_time_since_start.append(np.cumsum(one_seq_time)[-1])
        if max_time_since_last < max(one_seq_time):
            max_time_since_last = max(one_seq_time)
serialize_to_file((seq_event,seq_time),data_path)
print('the total sequences is {}'.format(len(seq_event)))
print('the longest seq length is {}'.format(max(seq_length)))
print('the longest time from start to end is {}'.format(max(max_time_since_start)))
print('the longest duration time is {}'.format(max_time_since_last))

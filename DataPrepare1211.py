import datetime
import json
import pickle
import os
import time
from title_replace_fun import replaceTitle

with open('/Users/qingxinmeng/PycharmProjects/time_sequence_prediction/source_data/c2i_1002.pkl','rb') as f:
    c2i=pickle.load(f)
with open('/Users/qingxinmeng/PycharmProjects/time_sequence_prediction/source_data/title2id.pkl','rb') as f:
    title2id=pickle.load(f)

dirpre='/Users/qingxinmeng/Documents/data_source/linkedin_data/linkedin3.0/link_target_no_des/part-0000'
paths=[dirpre+str(x) for x in range(10)]
person_set=set()
result = {}
n=0
for path in paths:
    n +=1
    start_time=time.time()
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        temp = line.strip().split(';')
        person_id=temp[0]
        temp1 = temp[1].split('.')
        career_seq = {}
        first_work_date='2019/3'
        company_num=set()
        if len(temp1)>4:
            for content in temp1:
                content = content.split(',')
                title=content[0]
                company=content[1]
                date = content[-1].split('-')
                start=date[0].strip()
                end=date[1].strip()
                if start != '' and end != '' and  company !='' and title !='' and start < end :
                    try:
                        if start < first_work_date:
                           first_work_date=start
                        if  (company in c2i) and start >'1988/1' and end < '2018/11':
                            if start not in career_seq:
                                normalized_title = replaceTitle(title)
                                if normalized_title !='':
                                    company_num.add(company)
                                    career_seq[start]={'company':[],'normalized_title':[],'end':[]}
                                    career_seq[start]['company'].append(company)
                                    career_seq[start]['normalized_title'].append(normalized_title)
                                    career_seq[start]['end'].append(end)
                    except ValueError:
                        continue
        if len(company_num)>3:
            seq = sorted(career_seq.items(), key=lambda x: x[0])
            person_set.add(person_id)
            result[person_id]={}
            result[person_id]['first_work_date']=first_work_date
            result[person_id]['company'] = [c2i[x[1]['company'][0]] for x in seq]
            result[person_id]['normalized_title']=[title2id[x[1]['normalized_title'][0]] for x in seq]
            result[person_id]['start'] = [x[0] for x in seq]
            result[person_id]['end'] = [x[1]['end'][0] for x in seq]
            durations = []
            working_months=[]
            f_year, f_month = result[person_id]['first_work_date'].split('/')
            for s, e  in zip(result[person_id]['start'], result[person_id]['end']):
                s_year,s_month = s.split('/')
                e_year,e_month = e.split('/')
                duration = int(e_year) * 12 + int(e_month) - int(s_year) * 12 - int(s_month)
                durations.append(duration)
                working_month=int(s_year) * 12 + int(s_month) - int(f_year) * 12 - int(f_month)
                working_months.append(working_month)
            result[person_id]['duration'] = durations
            result[person_id]['working_months']=working_months

    end_time=time.time()
    minutes=(end_time-start_time)/60
    print('for file:{} need time {:.2f}'.format(n,minutes))

json.dump(result, open('linkedin5.json', 'w'))

print(len(result))
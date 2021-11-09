import os
import pickle
from re import S
import numpy as np
import random
import argparse

def remove_ele(L, ele):
    return [e for e in L if e != ele]
def remove_list(L, l):
    return [e for e in L if e not in l]

parser = argparse.ArgumentParser()

parser.add_argument("--way", type=int, default=5, help="Number of classes per task.")
parser.add_argument("--shot", type=int, default=1, help="Number of samples per class")
parser.add_argument("--n_query", type=int, default=1, help="Number of query per task.")
parser.add_argument("--print", choices=["y", "n"], default="n", help="print result or not.")
args = parser.parse_args()


total_num=500
sample_dict_path = "./data/train_list.pkl"
save_dict_name = "train_task_list"
num_way = args.way
num_shot = args.shot
num_query = args.n_query

seed = 666
random.seed(seed)
with open(sample_dict_path, 'rb') as f:
    sample_dict = pickle.load(f)
task_list = {}
cls_list = list(sample_dict.keys())
cnt = 0
sample2id = {}
sample2cls = {}
for cls in cls_list:
    for sample in sample_dict[cls]:
        sample2id[sample] = cnt
        sample2cls[sample] = cls
        cnt+=1
# print(sample2id)
cnt = 0
while(cnt<total_num):
    sample_cls_list = random.sample(cls_list, num_way)
    query_cls = random.sample(sample_cls_list, num_query)

    support_sample=[]
    support_digit = []
    digit2cls = {}
    cls2digit = {}
    query_select_list = []
    # get support sample
    random.shuffle(sample_cls_list)
    for idx, cls in enumerate(sample_cls_list):
        support_sample.extend(random.sample(sample_dict[cls], num_shot))
        support_digit.append(idx)
        digit2cls[idx] = cls
        cls2digit[cls] = idx
        query_select_list.extend(sample_dict[cls])
    remove_list(query_select_list, support_sample)

    # calculate unique id of task
    query_digit = []
    query_sample=[]
    query_sample = random.sample(query_select_list, num_query)
    remove_list(query_select_list, query_sample)
    for s in query_sample:
        query_digit.append(cls2digit[sample2cls[s]])
    
    # get sample's id for non-repeat task
    all_sample_id = []
    for s in support_sample:
        all_sample_id.append(sample2id[s])
    for s in query_sample:
        all_sample_id.append(sample2id[s])
    all_sample_id.sort()
    for i in range(len(all_sample_id)):
        all_sample_id[i]=f"{all_sample_id[i]:04d}"
    task_id="".join(all_sample_id)
    # print(task_id)
    # add task
    if task_id not in task_list.keys():
        # task: Saved in the order of:
        #       support_sample, query_sample, support_digit, query_digit, digit2cls
        task_list[task_id]={"ss": support_sample, 
                            "qs": query_sample,
                            "sd": support_digit,
                            "qd": query_digit,
                            "d2c": digit2cls}
        cnt+=1
    else:
        print("Repeated task:", task_id, "cnt", cnt)

with open("./data/{}.pkl".format(save_dict_name), 'wb') as f:
    pickle.dump(task_list, f)

if(args.print=="y"):
    with open("./data/{}.pkl".format(save_dict_name), 'rb') as f:
        task_dict = pickle.load(f)
        for key, item in task_dict.items():
            print(item)


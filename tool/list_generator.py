from genericpath import samestat
import os
import sys
import pickle

dataset_sp = "test"
root = "/home/yanlq/datasets/UCF101"
sample_path = "{}/{}".format(root, dataset_sp)
sample_list = os.listdir(sample_path)
sample_list.sort()
sample_dict = {}

for cls in sample_list:
    samples = os.listdir("{}/{}".format(sample_path, cls))
    samples.sort()
    sample_dict[cls] = []
    for s in samples:
        sample_dict[cls].append("{}/{}/{}".format(sample_path, cls, s))

with open("data/{}_list.pkl".format(dataset_sp), 'wb') as f:
    pickle.dump(sample_dict, f)
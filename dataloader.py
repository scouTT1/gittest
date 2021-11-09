import os
import torch
import cv2
import pickle
import numpy as np
import random
from torch.nn.functional import smooth_l1_loss
from torch.utils.data import Dataset, DataLoader
from transforms import VideoTransform


def remove_ele(L, ele):
    return [e for e in L if e != ele]


def remove_list(L, l):
    return [e for e in L if e not in l]


class MetaDataset(Dataset):
    def __init__(self, task_dict, num_frame, is_trans, seed=1):
        with open(task_dict, 'rb') as f:
            self.task_dict = pickle.load(f)
        self.task_ids = list(self.task_dict.keys())
        # self.num_way = num_way
        # self.num_shot = num_shot
        self.num_way = len(self.task_dict[self.task_ids[0]]['sd'])
        self.num_shot = int(len(self.task_dict[self.task_ids[0]]['ss']) / self.num_way)
        self.is_trans = is_trans
        self.num_frame = num_frame
        self.trans = VideoTransform()

        # generate task list

    def __getitem__(self, index):
        ## every task has (N*K+K) samples
        task_id = self.task_ids[index]
        # print(task_id)
        support_samples = self.consecutive_load_vedio(self.task_dict[task_id]["ss"])
        query_samples = self.consecutive_load_vedio(self.task_dict[task_id]["qs"])

        return support_samples, torch.as_tensor(self.task_dict[task_id]["sd"]), \
               query_samples, torch.as_tensor(self.task_dict[task_id]["qd"]), \
               self.task_dict[task_id]["d2c"]

    def __len__(self):
        return len(self.task_ids)

    def consecutive_load_vedio(self, sample_paths):
        samples = []
        for path in sample_paths:
            frame_list = os.listdir(path)
            frame_list.sort()
            start_frame = random.choice(range(len(frame_list) - self.num_frame))
            sample = []
            for i in range(self.num_frame):
                image = cv2.imread("{}/{}".format(path, frame_list[i + start_frame]))
                sample.append(image)

            if self.is_trans:
                sample = self.trans(torch.as_tensor(sample))
            samples.append(sample)
        samples = torch.stack(samples)
        return samples

    def uniform_load_video(self, sample_paths):
        samples = []
        for path in sample_paths:
            frame_list = os.listdir(path)
            frame_list.sort()
            len_video = len(frame_list)
            frame_gap = len_video // self.num_frame
            start_frame = random.choice(range((len_video - 1) % self.num_frame + 1))
            sample = []
            for i in range(self.num_frame):
                image = cv2.imread("{}/{}".format(path, frame_list[i * frame_gap + start_frame]))
                sample.append(image)

            if self.is_trans:
                sample = self.trans(torch.as_tensor(sample))
            samples.append(sample)
        samples = torch.stack(samples)
        return samples
# example  
# dataset = MetaDataset("./data/valid_list.pkl", 
#                       num_way=5,
#                       num_shot=1,
#                       num_frame=16,
#                       is_trans=False,
#                       seed=666)
# loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: x)
# print(dataset.__len__())
# for idx, sample in enumerate(loader):
#     support_sample, support_digit, query_sample, query_digit, digit2label = sample[0]
#     print(idx, support_sample.shape, support_digit.shape, 
#                query_sample.shape, query_digit.shape, 
#                digit2label)
#     break
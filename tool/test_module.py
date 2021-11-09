import os, sys
import numpy as np
import six

import string
import lmdb
import pickle
import cv2
# import msgpack
import tqdm
import pyarrow as pa
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
# from torchvision.datasets import ImageFolder
import dataloader
from c3d import C3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = dataloader.MetaDataset("./data/train_task_list.pkl", num_way=5, num_shot=1, num_frame=16, is_trans=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4, collate_fn=lambda x: x)
c3d=C3D()
c3d.load_state_dict(torch.load('./ucf101-caffe.pth'))
c3d = c3d.to(device)
c3d.eval()
cnt=0
for datas in train_loader:
    support_video, support_labels, target_video, target_labels, digit2cls = datas[0]
    target_video.permute(0,2,3,4,1)
    target_video=target_video.to(device)
    predict = c3d(target_video)
    print(digit2cls[int(target_labels[0])], torch.argmax(predict))
    cnt+=1
    if(cnt>2):
        break

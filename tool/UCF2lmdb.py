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

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
# from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

def ucf2lmdb(dpath, name="train", write_frequency=500, num_workers=8):
    # directory = os.path.expanduser(os.path.join(dpath, name))
    directory = os.path.expanduser(os.path.join(dpath, name))
    print("Loading data from %s." % directory)

    lmdb_path = os.path.expanduser(os.path.join(dpath, '{}.lmdb'.format(name)))
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s." % lmdb_path) 
    video_list = os.listdir(directory)
    video_list.sort()
    cls_index_list = {}
    # calculate size
    size = 0
    for _, video in enumerate(video_list):
        # prepare
        vpath = os.path.join(directory, video)
        cap = cv2.VideoCapture(vpath)
        framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if framenum<16:
            continue
        size+=240*320*3*int(framenum)
    print("Create %s byte lmdb file." % size)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=size*8/1024+1048576, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    idx = 0
    for _, video in enumerate(video_list):
        # prepare
        vpath = os.path.join(directory, video)
        cap = cv2.VideoCapture(vpath)
        framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if framenum<16:
            continue
        
        # dump video frames
        frames = []
        for i in range(framenum):
            ret, image = cap.read()
            frames.append(image)
        frames = np.array(frames)

        label = video.split('_')[1]
        if label not in cls_index_list:
            cls_index_list[label] = []
        cls_index_list[label].append(idx)

        # write
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((frames, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(video_list)))
            txn.commit()
            txn = db.begin(write=True)
        idx = idx + 1

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
        
    # save class index dict
    with open("data/{}_cls_index_list.pkl".format(name), 'wb') as f:
        pickle.dump(cls_index_list, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="valid")
    parser.add_argument('--out', type=str, default=".")
    # parser.add_argument('--size', type=float, default=7.0)
    parser.add_argument('-p', '--procs', type=int, default=4)

    args = parser.parse_args()

    ucf2lmdb(args.folder, name=args.split, num_workers=args.procs)
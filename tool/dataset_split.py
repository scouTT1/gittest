import os
import sys
import pickle

data_root = "/home/yanlq/datasets/UCF101"
videolist = os.listdir(data_root)
videolist.sort()
if not os.path.exists("{}/train".format(data_root)):
    os.system("mkdir {}/train".format(data_root))
if not os.path.exists("{}/valid".format(data_root)):
    os.system("mkdir {}/valid".format(data_root))
if not os.path.exists("{}/test".format(data_root)):
    os.system("mkdir {}/test".format(data_root))
cnt=0
pre_class=""
now_class=""
sp = "train"
for v in videolist:
    now_class = v.split('_')[1]

    if pre_class != now_class:
        ratio = cnt*1.0/len(videolist)
        if ratio < 0.7:
            sp = "train"
            # if not os.path.exists("{}/train/{}".format(data_root, now_class)):
            #     os.system("mkdir {}/train/{}".format(data_root, now_class))
            print("train:", now_class)
        elif ratio < 0.8:
            sp = "valid"
            # if not os.path.exists("{}/valid/{}".format(data_root, now_class)):
            #     os.system("mkdir {}/valid/{}".format(data_root, now_class))
            print("valid:", now_class)
        else:
            sp = "test"
            # if not os.path.exists("{}/test/{}".format(data_root, now_class)):
            #     os.system("mkdir {}/test/{}".format(data_root, now_class))
            print("test:", now_class)

    old_path = "{}/{}".format(data_root, v)
    # new_path = "{}/{}/{}/{}".format(data_root, sp, now_class, v)
    new_path = "{}/{}/{}".format(data_root, sp, v)
    os.system("mv {} {}".format(old_path, new_path))
    pre_class = now_class
    cnt += 1

# print(len(videolist))
import argparse
import os
import sys
import cv2

parser = argparse.ArgumentParser(description='video to frame.')
parser.add_argument('--sp', help='specify it is train, valid or test split.')
args = parser.parse_args()
dataset_sp = args.sp
root = "/home/yanlq/datasets/UCF101"
sample_path = "{}/{}".format(root, dataset_sp)
sample_list = os.listdir(sample_path)
sample_list.sort()
sample_dict = {}

for cls in sample_list:
    cls_path = "{}/{}".format(sample_path, cls)
    path_list = os.listdir(cls_path)
    path_list.sort() 
    for path in path_list:
        video = "{}/{}".format(cls_path, path)
        save_path = video.split('.')[0]
        if not os.path.exists(save_path):
            os.system("mkdir {}".format(save_path))
        vc = cv2.VideoCapture(video)
        if vc.get(cv2.CAP_PROP_FRAME_COUNT) < 16:
            print(video, "less than 16 frames")
            continue
        cnt = 1
        while True:
            ret, frame = vc.read()
            # store operation every frame
            if ret:
                frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)
                cv2.imwrite("{}/{}.BMP".format(save_path, cnt), frame)
            else:
                break
            cnt+=1
        vc.release()
        print(save_path)

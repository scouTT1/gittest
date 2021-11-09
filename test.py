from numpy.core.records import array
import torch
import numpy as np
import argparse
import os
import pickle

import torch.nn.functional as F

from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, \
    task_confusion
from model import CNN_TRX

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
#import tensorflow as tf

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import dataloader
import random


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        ckpt_paths = os.listdir(self.args.test_ckpt_dir)
        for p in ckpt_paths:
            if p.split('.')[-1] == 'txt':
                self.args.logfile_path = os.path.join(self.args.test_ckpt_dir, p)
            if p.split('.')[-1] == 'pt':
                self.args.test_ckpt = os.path.join(self.args.test_ckpt_dir, p)
        if os.path.isfile(self.args.logfile_path):
            self.logfile = open(self.args.logfile_path, "a", buffering=1)
        else:
            self.logfile = open(self.args.logfile_path, "w", buffering=1)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.args.checkpoint_dir)
        print_and_log(self.logfile, "Test Mode.\n")

        # self.writer = SummaryWriter()

        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.test_set = self.init_data()

        self.test_dataset = dataloader.MetaDataset(os.path.join(self.args.split_dir, "test_task_list.pkl"),
                                                   num_frame=self.args.frame_num, is_trans=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1,
                                                       num_workers=self.args.num_workers, collate_fn=lambda x: x)

        self.loss = torch.nn.CrossEntropyLoss()
        # self.loss = torch.nn.MSELoss()
        self.accuracy_fn = aggregate_accuracy

    def init_model(self):
        model = CNN_TRX(self.args)
        checkpoint = torch.load(self.args.test_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        with open(os.path.join(self.args.split_dir, "test_task_list.pkl"), 'rb') as f:
            test_set = pickle.load(f)
        return test_set

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf"], default="ucf",
                            help="Dataset to use.")
        parser.add_argument("--split_dir", type=str, help="Directory to data splits.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.002, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=0,
                            help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5,
                            help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1,
                            help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int,
                            help='iterations to test at. Default is for ssv2 otam split.', default=[75000])
        parser.add_argument("--print_freq", type=int, default=5000, help="print and log every n iterations.")
        parser.add_argument("--frame_num", type=int, default=16, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=4, help="Num dataloader workers.")
        parser.add_argument("--method", choices=["c3d", "resnet"], default="c3d", help="method")
        parser.add_argument("--trans_linear_out_dim", type=int, default=512, help="Transformer linear_out_dim")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="adam", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=10000,
                            help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--scratch", choices=["bc", "bp"], default="bc",
                            help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true",
                            help="Load 1 vid per class for debugging")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[10000])

        args = parser.parse_args()

        if args.scratch == "bc":
            args.scratch = "/mnt/storage/home/tp8961/scratch"
        elif args.scratch == "bp":
            args.num_gpus = 3
            # this is low becuase of RAM constraints for the data loader
            args.num_workers = 6
            args.scratch = "/work/tp8961"

        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if (args.method == "c3d"):
            args.img_size = 112
            args.trans_linear_in_dim = 512
            args.trans_linear_out_dim = 1152
        elif (args.method == "resnet"):
            args.img_size = 224
            args.trans_linear_in_dim = 512
            args.trans_linear_out_dim = 1152
        return args

    def run(self):
        losses = []
        accuracies = []
        iteration = 0
        self.model.eval()
        with torch.no_grad():
            accuracies = []
            iteration = 0
            for task_dict in self.test_loader:
                iteration += 1

                context_images, context_labels, target_images, target_labels, digit2cls = task_dict[0]
                context_images = context_images.to(self.device)
                target_images = target_images.to(self.device)
                context_labels = context_labels.to(self.device)
                target_labels = target_labels.type(torch.LongTensor).to(self.device)

                # forward
                target_logits = self.model(context_images, context_labels, target_images)
                task_loss = self.loss(target_logits, target_labels)
                task_accuracy = self.accuracy_fn(target_logits, target_labels)

                print("loss:", task_loss.item(), "acc:", task_accuracy.item())
                losses.append(task_loss.item())
                accuracies.append(task_accuracy.item())

            ave_loss = np.array(losses).mean()
            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            # print training stats
            print_and_log(self.logfile, "{0:.6f}: {1:.6f}+/-{2:.6f}".format(ave_loss, accuracy, confidence))
            print_and_log(self.logfile, "")  # add a blank line

        self.logfile.close()


if __name__ == "__main__":
    learner = Learner()
    learner.run()

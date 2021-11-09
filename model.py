from typing import no_type_check_decorator
import torch
import torch.nn as nn
from collections import OrderedDict

from torch.nn.utils.rnn import pad_packed_sequence
from utils import split_first_dim_linear, task_confusion
import math
from itertools import combinations

from torch.autograd import Variable

import torchvision.models as models
from c3d import C3D

NUM_SAMPLES = 1


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=100, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, seq_len):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args

        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.sk_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)  # .cuda()
        self.sv_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)  # .cuda()
        self.qk_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)  # .cuda()
        self.qv_linear = nn.Linear(self.args.trans_linear_in_dim, self.args.trans_linear_out_dim)  # .cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=1)

    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        len_local = support_set.shape[1]
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # apply linear maps
        support_set_ks = self.sk_linear(support_set)
        queries_ks = self.qk_linear(queries)
        support_set_vs = self.sv_linear(support_set)
        queries_vs = self.qv_linear(queries)

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs

        unique_labels = torch.unique(support_labels)  # 挑出不重复的label

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way).cuda()

        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)  # (n_query,n_support,L,d_k)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)
            class_scores = class_scores.reshape(n_queries, len_local, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, len_local, -1, len_local)
            class_scores = class_scores.permute(0, 2, 1, 3)

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores,
                                           class_v)  # matmul([1,n_query,n_support], [1,n_support,dim_feature])=[1,n_query,dim_feature]
            query_prototype = torch.sum(query_prototype, dim=1)

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            distance = torch.norm(diff, dim=[-2, -1]) ** 2
            distance = torch.div(distance, math.sqrt(self.args.trans_linear_out_dim * len_local))

            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance

        return all_distances_tensor

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class CNN_TRX(nn.Module):
    """
    Standard Resnet connected to a Temporal Cross Transformer.

    """

    def __init__(self, args):
        super(CNN_TRX, self).__init__()

        self.train()
        self.args = args

        if self.args.method == "c3d":
            c3d = C3D()
            c3d.load_state_dict(torch.load("./ucf101-caffe.pth"))
            last_layer_idx = -1
            self.c3d = nn.Sequential(*list(c3d.children())[:last_layer_idx])  # output: [512, 1, 4, 4]
            # for name, param in c3d.named_parameters():
            #     param.requires_grad=False
        elif self.args.method == "resnet":
            resnet = models.resnet50(pretrained=True)
            last_layer_idx = -1
            self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])  # output: [n, 2048]
            self.hidden_size = 512
            self.max_length = 80
            self.n_layers = 1
            self.rnn = nn.RNN(2048, self.hidden_size, self.n_layers, batch_first=True)
            # for name, param in self.resnet.named_parameters():
            #     param.requires_grad=False
        elif self.args.method == "none":
            self.fc = nn.Sequential(nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512))

        # self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])
        self.transformers = TemporalCrossTransformer(args, 100)

    def forward(self, context_images, context_labels, target_images, target_labels):
        n_supports = context_labels.shape[0]
        n_queries = target_labels.shape[0]
        if self.args.method == "c3d":
            context_features = self.embedding(context_images).view(n_supports, 512, 16)  # (n_supports, 512, 16)
            context_features = context_features.permute(0, 2, 1)  # (n_supports, 16, 512)
            target_features = self.embedding(target_images).view(n_queries, 512, 16)  # (n_queries, 512, 16)
            target_features = target_features.permute(0, 2, 1)  # (n_queries, 16, 512)
        elif self.args.method == "resnet":
            context_images = context_images.permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)
            target_images = target_images.permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)
            context_features = self.resnet(context_images)
            target_features = self.resnet(target_images)

            context_features = context_features.reshape(n_supports, -1, 2048)
            h0 = Variable(torch.randn(self.n_layers, n_supports, self.hidden_size)).cuda(0)
            context_features, _ = self.rnn(context_features, h0)
            # context_features = pad_packed_sequence(out)[0]

            target_features = target_features.reshape(n_queries, -1, 2048)
            h0 = Variable(torch.randn(self.n_layers, n_queries, self.hidden_size)).cuda(0)
            target_features, _ = self.rnn(target_features, h0)
            # target_features = pad_packed_sequence(out)[0]
        elif self.args.method == "none":
            context_features = context_images
            target_features = target_images
            # context_features = self.fc(context_features)
            # target_features = self.fc(target_features)
        all_logits = self.transformers(context_features, context_labels, target_features)

        return all_logits

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.method == "c3d":
            if self.args.num_gpus > 1:
                self.c3d.cuda(0)
                self.c3d = torch.nn.DataParallel(self.c3d, device_ids=[i for i in range(0, self.args.num_gpus)])

                self.transformers.cuda(0)
        if self.args.method == "resnet":
            if self.args.num_gpus > 1:
                self.resnet.cuda(0)
                self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])
                self.rnn.cuda(0)
                self.transformers.cuda(0)
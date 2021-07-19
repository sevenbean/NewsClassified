'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/6/29_17:22
    @filename:
    @description:
'''
import os

import torch
import torch.nn.functional as F
from torch import nn




# in_channels 就是词向量的维度, out_channels则是卷积核(每个kernel_size都一样多)的数量
# 对于一种尺寸的卷积核  kernel_size = 3,in_channels=100,out_channels=50时,
# 设 x = [32][100]即一个新闻数据, 进行卷积操作前，先对x维度进行变换->[100][32]，即每一列是一个词向量，conv1d卷积层左右扫描即可
# x经过卷积操作后会得到[50][30], 分别对应50个卷积核分别对x运算得到的一维数据的结果,即[out_channels][in_channels-kernel_size + 1],
# 然后进行 relu运算，形状不变
# 紧接着进行最大池化操作,每个卷积核运算结果中选出一个最大值,即[30]中选出一个, -> max = [50][1]
# 接着将max转为一维数据 ->[50], 即[out_channels]
# 由于有len(kernel_size)种尺寸的卷积核,所以经过卷积层，池化层有 len(kernel_size) * output_channels 个输出
class TextCNN1d(nn.Module):
    def __init__(self,opt,embedding_matrix):
        super(TextCNN1d, self).__init__()
        self.opt=opt
        self.kernel_size =(3,4,5)
        self.output_channels = opt.output_channels
        self.class_num = opt.class_num
        self.liner_one = opt.linear_one
        self.dropout = opt.dropout
        # 也可以这么写，不用调用 init_embedding方法，利用已有的word2vec预训练模型中初始化
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix,dtype=torch.float))
        # self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.vector_size, padding_idx=0)
        # 这里是不同kernel_size的conv1d
        self.convs = [nn.Conv1d(in_channels=self.embedding.embedding_dim, out_channels=self.output_channels,
                                kernel_size=size, stride=1, padding=0).to(self.opt.device)
                      for size in self.kernel_size]
        self.fc1 = nn.Linear(len(self.kernel_size) * self.output_channels, self.liner_one)
        self.fc2 = nn.Linear(self.liner_one, self.class_num)
        self.dropout = nn.Dropout(self.dropout)

    # embedding_matrix就是word2vec的get_vector_weight
    def init_embedding(self, embedding_matrix):
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix).to(self.opt.device))

    # x = torch.tensor([[word1_index, word2_index,..],[],[]])
    # forward这个函数定义了前向传播的运算
    def forward(self, x):
        x = self.embedding(x)  # 词索引经过词嵌入层转化为词向量, [word_index1,word_index2]->[[vector1][vector2]],
        x = x.permute(0, 2, 1)  # 将(news_num, words_num, vector_size)换为(news_num,vector_size,word_num),方便卷积层运算
        # 将所有经过卷积、池化的结果拼接在一起
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        # 展开,[news_num][..]
        x = x.view(-1, len(self.kernel_size) * self.output_channels)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2))  # 最大池化
        x = x.squeeze(2)  # 只保存2维
        return x

class TextCNN(nn.Module):
    def __init__(self,opt,embedding_matrix):
        # n_filter 每个卷积核的个数
        super(TextCNN, self).__init__()
        self.opt = opt
        self.kernel_size = (3,4,5)
        self.output_channels = opt.output_channels
        self.class_num = opt.class_num
        self.dropout = opt.dropout

        # 也可以这么写，不用调用 init_embedding方法，利用已有的word2vec预训练模型中初始化
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.output_channels, kernel_size=(fs, self.opt.embed_dim)) for fs in
             self.kernel_size])
        self.fc = nn.Linear(len(self.kernel_size) * self.output_channels,self.class_num)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1).float()
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, int(conv.shape[2])).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        # cat = self.fc(cat)
        # cat = cat.unsqueeze(dim=-1)
        return self.fc(cat)
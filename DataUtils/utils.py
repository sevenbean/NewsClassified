'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/6/28_15:59
    @filename:
    @description:
'''
import re
import pandas as pd
from tqdm import tqdm
import pickle
import jieba
import os
import argparse
from gensim.models import KeyedVectors,word2vec
from time import time
import numpy as np

# 正则化清洗函数
def clear_data(text):
    r1 = r"[^\u4e00-\u9fa5a-zA-Z0-9]"
    text = re.sub(r1, '', str(text))
    return text

def read_stopword(fpath):
    """
    读取中文停用词表
    """
    with open(fpath, 'r', encoding='utf-8') as file:
        stopword = file.readlines()
    return [word.replace('\n', '') for word in stopword]

# 单句分词
def cut_sentence(sentence):
    # 加载stop_words停用词
    # stop_words = pd.read_table(args.stopwords_file, header=None)[0].tolist()
    tokens = list(jieba.cut(sentence))
    # 去除停用词  列表
    # tokens = [token for token in tokens]
    # name_list = os.listdir(stopwords_file)
    # stop_word = []
    # for fname in name_list:
    #     stop_word += read_stopword(os.path.join(stopwords_file, fname))
    # stop_words = set(stop_word)  # 停用词表去重
    # tokens = [token for token in tokens if token not in stop_words]
    return tokens

#处理训练数据
def process_data(data_filepath,cut_filepath,mode='title_content'):

    if(os.path.exists(cut_filepath)):
        return pickle.load(open(cut_filepath,"rb"))
    else:
        # df数据

        data = pd.read_csv(data_filepath)
        # 应用正则化清洗和分词函数
        def apply_data(data, mode):
            # 读取数据 - 清洗（去字符，去重） - 分词
            if 'title' not in data.columns.tolist() or 'content' not in data.columns.tolist():
                return "csv文件中没有对应的列名，请确认是否包含'title'、'content'、'label'!"
            elif mode == 'title_content':
                data['title'] = data['title'].apply(lambda x: clear_data(str(x)))
                data['content'] = data['content'].apply(lambda x: clear_data(x))
                data = data.drop_duplicates(subset=None, keep='first', inplace=False).reset_index(drop=True)

                print("分词中...")
                train_set = [cut_sentence(x) for x in tqdm((data['title'] + data['content']).values.tolist())]
            else:
                data[mode] = data[mode].apply(lambda x: clear_data(str(x)))
                data = data.drop_duplicates(subset=mode, keep='first', inplace=False).reset_index(drop=True)

                print("分词中...")
                #       train_set = data[text_part].apply(lambda x: cut_sentence(x))
                train_set = [cut_sentence(x) for x in tqdm(data[mode].values.tolist())]

            # 保存分词结果
            with open(cut_filepath, 'wb') as f:
                pickle.dump(train_set, f)

            return train_set
        # 返回
        return apply_data(data,mode)

#利用训练集构建词典
def build_vocab(train_set,args):
    print("--构建词典中——")
    start_time=time()
    model = word2vec.Word2Vec(train_set  # 将中文句子进行分词
                                  , sg=0  # 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法；
                                  , size=args.embed_dim  # 是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百；
                                  , window=5  # 表示当前词与预测词在一个句子中的最大距离是多少
                                  , min_count=5  # 词频少于min_count次数的单词会被丢弃掉, 默认值为5；
                                  , negative=3  # 如果>0,则会采用negative samping，用于设置多少个noise words
                                  , sample=0.001  # 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
                                  , hs=0  # 如果为1则会采用hierarchical softmax技巧。如果设置为0（defaut），则negative sampling会被使用；
                                  , workers=4  # 参数控制训练的并行数；
                                  )
    model.wv.save_word2vec_format(args.word2vec_filepath)
    print("词向量模型训练结束共消耗：{}".format(time()-start_time))
    return model.wv

class Tokenizer(object):
    def __init__(self,opt):
        super(Tokenizer,self).__init__()
        self.word2id={}
        self.id2word={}
        self.idx=1
        self.opt=opt
        self.train_set=process_data(opt.train_data_filepath,opt.train_cut_data)

    #加载已经训练好的词向量
    def load_wordvector(self):
        print("------------load wordvec-----------")
        if (os.path.exists(self.opt.word2vec_filepath)):
            wordvec = KeyedVectors.load(self.opt.word2vec_filepath)
        else:
            wordvec=build_vocab(self.train_set,self.opt)
        return wordvec

    #文本数据的填充和剪切
    def pad_and_truncat(self,sequence,padding="post", truncating="post", dtype="int64",value=0):
        x = (np.ones(self.opt.max_seq_len) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-self.opt.max_seq_len:]
        else:
            trunc = sequence[:self.opt.max_seq_len]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    #创建该语料的word2idx和idx2word
    def fit_on_text(self):
        # word_freq={}
        for word_list in self.train_set:
            for word in word_list:
                # word_freq[word]=word_freq.get(word,0)+1
                if word not in self.word2id:
                            self.word2id[word]=self.idx
                            self.id2word[self.idx]=word
                            self.idx+=1
        # word_vocab=[word for word in word_freq.keys() if word_freq[word]>self.opt.min_word_freq]
        #
        # self.word2idx = {k: v + 1 for v, k in enumerate(word_vocab)}
        # self.id2word={v+1:k for v,k in enumerate(word_vocab)}


    #将文本数据转为对应的数字序列
    def text_to_sequence(self,train_set,padding='post', truncating='post'):
        dataSet=[]
        for word_list in train_set:
            sequences = [self.word2id[word] if word in self.word2id.keys() else 0 for word in word_list]
            if len(sequences)==0:
                sequences=[0]
            dataSet.append(self.pad_and_truncat(sequences,padding=padding, truncating=truncating))
        return dataSet

    #将序列转为文本，以验证文本转为数字序列是否正确
    def sequence_to_text(self,text):
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        word_list = [word for word in jieba.cut(text)]
        sequences = [self.word2id[word] for word in word_list]
        text=[self.id2word[idx] for idx in sequences]
        return text

    #构建词向量矩阵
    def build_embeding_matrix(self):
        if (os.path.exists(self.opt.embedding_matrix_filepath)):
            print("loading_embedding_matrix:", self.opt.embedding_matrix_filepath)
            embedding_matrix = pickle.load(open(self.opt.embedding_matrix_filepath, "rb"))
        else:
            print("loading word vector.......")
            embedding_matrix = np.zeros((len(self.word2id) + 1, self.opt.embed_dim))
            wordvec = self.load_wordvector()
            for word, index in self.word2id.items():
                if word in wordvec.vocab.keys():
                    embedding_matrix[index] = wordvec.wv[word]
            pickle.dump(embedding_matrix, open(self.opt.embedding_matrix_filepath, "wb"))
        return embedding_matrix


def build_tokenizer(opt):
    if(os.path.exists(opt.tokenizer_path)):
        tokenizer=pickle.load(open(opt.tokenizer_path,"rb"))
    else:
        tokenizer=Tokenizer(opt)
        tokenizer.fit_on_text()
        pickle.dump(tokenizer,open(opt.tokenizer_path,"wb"))
    return tokenizer




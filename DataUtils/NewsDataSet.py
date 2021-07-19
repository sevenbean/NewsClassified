'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/6/29_15:23
    @filename:
    @description:
'''
from torch.utils.data import Dataset
from DataUtils.utils import process_data,clear_data
import pandas as pd
import torch
class NewsDataSet(Dataset):
    def __init__(self,tokenizer):
        super(NewsDataSet, self).__init__()
        self.opt=tokenizer.opt
        data = pd.read_csv(tokenizer.opt.train_data_filepath)
        self.tokens_id=torch.LongTensor(tokenizer.text_to_sequence(process_data(self.opt.train_data_filepath
                                                                                ,self.opt.train_cut_data)))
        self.y=data["label"].values.tolist()


    def __getitem__(self, item):
       return self.tokens_id[item],self.y[item]

    def __len__(self):
        return len(self.tokens_id)

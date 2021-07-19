'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/7/13_0:11
    @filename:
    @description:
'''
from torch.utils.data import Dataset
from DataUtils.utils import clear_data,cut_sentence

from tqdm import tqdm
import torch

def process_data(data,mode='title_content'):
        def apply_data(data, mode):
            # 读取数据 - 清洗（去字符，去重） - 分词
            if 'title' not in data.columns.tolist() or 'content' not in data.columns.tolist():
                return "csv文件中没有对应的列名，请确认是否包含'title'、'content'、'label'!"
            elif mode=="title_content":
                data['title'] = data['title'].apply(lambda x: clear_data(str(x)))
                data['content'] = data['content'].apply(lambda x: clear_data(x))
                data = data.drop_duplicates(subset=None, keep='first', inplace=False).reset_index(drop=True)

                print("分词中...")
                train_set = [cut_sentence(x) for x in tqdm((data['title'] + data['content']).values.tolist())]
            return train_set
        # 返回
        return apply_data(data,mode)

class pre_DataSet(Dataset):
    def __init__(self,tokenizer,data=None):
        super(pre_DataSet, self).__init__()
        self.opt=tokenizer.opt
        self.tokens_id=torch.LongTensor(tokenizer.text_to_sequence(process_data(data))).to(tokenizer.opt.device)

    def __getitem__(self, item):
        return self.tokens_id[item]

    def __len__(self):
        return len(self.tokens_id)

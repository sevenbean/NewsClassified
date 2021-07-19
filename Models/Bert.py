'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/7/9_13:10
    @filename:
    @description:
'''

from transformers import BertModel,BertConfig
import torch.nn as nn

class Bert(nn.Module):
    def __init__(self,opt):
        super(Bert, self).__init__()

        self.config=BertConfig.from_pretrained(r"./bert-base-chinese/bert_config.json")
        self.bert=BertModel.from_pretrained(r"./bert-base-chinese/pytorch_model.bin",config=self.config)

        self.config.output_attentions=True
        self.config.output_hidden_states=True

        self.fc=nn.Linear(self.config.hidden_size,opt.class_num)

    def forward(self,input_ids,attention_mask):

        outputs=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        pooler_putput=outputs[1]
        out=self.fc(pooler_putput)
        return out
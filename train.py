'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/6/29_16:08
    @filename:
    @description:
'''
import argparse
from DataUtils.utils import build_tokenizer
from DataUtils.NewsDataSet import  NewsDataSet
import torch
from Models.TextCNN import TextCNN1d,TextCNN
from Models.DPCNN import DPCNN

from Models.Bert import Bert
from torch.utils.data import random_split,DataLoader
import logging
import os
import sys
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy
import random
from transformers import BertTokenizer

logger=logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Construction(object):
    def __init__(self,opt):
        super(Construction, self).__init__()
        self.opt=opt
        self.tokenizer=build_tokenizer(opt)
        self.embedding_matrix = self.tokenizer.build_embeding_matrix()
        self.model = self.opt.model_class(opt, self.embedding_matrix).to(self.opt.device)
        self.newsDataSet=NewsDataSet(self.tokenizer)

        if(opt.test_size>0):
            valid_radio=int(len(self.newsDataSet)*opt.test_size)
            self.train_DataSet,self.val_DataSet=random_split(self.newsDataSet, (len(self.newsDataSet) - valid_radio, valid_radio))

        self.target_names = ['体育', '其他','军事', '娱乐', '房产', '教育', '汽车', '游戏', '科技', '财经']

    def _train(self,train_DataLoader,val_DataLoader,optimizer,criterion):

        max_test_f1 = 0
        max_test_acc=0
        global_step = 0
        max_epoch=0
        for epoch in range(self.opt.num_epoch):
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for t_batchsize, inputs in enumerate(train_DataLoader):
                inputs = tuple(t.to(self.opt.device) for t in inputs)
                global_step += 1
                optimizer.zero_grad()
                train_x,train_y=inputs
                outputs=self.model(train_x)

                loss = criterion(outputs,train_y)
                loss.backward()
                optimizer.step()
                # 正确的数量
                y_pred=torch.argmax(outputs, -1)
                n_correct += (y_pred== train_y).sum().item()
                # 总数
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % 5 == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total

                    logger.info('{}----> loss: {:.4f}, acc: {:.4f}'.format(epoch+1,train_loss, train_acc))
                    # 验证集评估

            val_acc, val_f1 = self._evaluate_acc_f1(val_DataLoader)

            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_f1 >= max_test_f1:
                max_test_f1 = val_f1
                max_test_acc=val_acc
                max_epoch=epoch+1
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                # 保存model
                torch.save(self.model,self.opt.save_model_path)
        print("{}_best_result:----epoch:{}--------------accuracy:{}----------------f1:{}".format(self.opt.model_name,max_epoch,max_test_acc,max_test_f1))

    def _evaluate_acc_f1(self,data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch Mymodel to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, inputs in enumerate(data_loader):
                inputs = tuple(t.to(self.opt.device) for t in inputs)
                val_x,val_y=inputs
                t_outputs = self.model(val_x)

                y_pred=torch.argmax(t_outputs, -1)
                n_correct += (y_pred == val_y).sum().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = val_y
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, val_y), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = metrics.accuracy_score(y_true=t_targets_all.cpu(),y_pred=torch.argmax(t_outputs_all,1).cpu())
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),average="macro")
        report=classification_report(t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu(),labels=[0,1,2,3,4,5,6,7,8,9],target_names=self.target_names)
        print(report)
        return acc, f1


    def _run(self):
        train_DataLoader=DataLoader(self.train_DataSet,batch_size=self.opt.batch_size)
        val_DataLoader=DataLoader(self.val_DataSet,batch_size=self.opt.batch_size)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=self.opt.learning_rate)
        criterion=torch.nn.CrossEntropyLoss()
        self._train(train_DataLoader,val_DataLoader,optimizer,criterion)


def main():
    train_data_name="clear_without_noise_train10k"

    model_name="TextCNN"
    args = argparse.ArgumentParser()
    args.add_argument("--train_cut_data", default="./DataSet/cutdata/{}_without_stopwords_cut_words.dat".format(train_data_name), type=str)

    args.add_argument("--train_data_filepath", default="./DataSet/{}.csv".format(train_data_name), type=str)
    args.add_argument("--word2vec_filepath", default="./word2vec/Tencent_ChineseEmbedding.bin", type=str)
    args.add_argument("--embedding_matrix_filepath", default="./word2vec/{}_{}_without_stopwords_embedding_matrix.dat".format(model_name,train_data_name), type=str)
    args.add_argument("--max_seq_len", default=512, type=int)
    args.add_argument("--embed_dim", default=200, type=int)

    args.add_argument("--tokenizer_path",default="./word2vec/{}_{}_without_stopwords_tokenizer.dat".format(model_name,train_data_name),type=str)
    args.add_argument("--test_size",default=0.3,type=float)
    args.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    args.add_argument("--model_name",default=model_name,type=str)
    args.add_argument("--batch_size", default=16, type=int)
    args.add_argument("--learning_rate", default=0.002, type=float)
    args.add_argument("--num_epoch", default=20, type=int)
    args.add_argument("--seed",default=13,type=int)

    args.add_argument("--save_model_path",default="./state_dict/{}_best_model_without_stopwords_model_512.pth".format(model_name))

    #TextCNN中的参数
    args.add_argument("--dropout",default=0.5,type=float)
    args.add_argument("--class_num",default=10,type=int)
    args.add_argument("--output_channels",default=100,type=int)# 每种尺寸的卷积核有多少个
    args.add_argument("--linear_one",default=250)
    #DPCNN中的参数
    args.add_argument("--num_filters", default=250, type=int)

    model_classes={"TextCNN":TextCNN
                 ,"TextCNN1d":TextCNN1d
                    ,"bert":Bert
                   ,"DPCNN":DPCNN}

    opt = args.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    opt.model_class=model_classes[model_name]

    construct=Construction(opt)
    construct._run()


if __name__ == '__main__':
    main()
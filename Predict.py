'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/7/11_13:29
    @filename:
    @description:
'''
import torch
from Models.TextCNN import TextCNN
from DataUtils.utils import *
import pickle
from time import time
from DataUtils.pre_Dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def predcit(text):
    print("加载tokenizer")
    tokenizer=pickle.load(open("./word2vec/TextCNN_clear_without_noise_train10k_tokenizer.dat","rb"))
    print("加载词典")
    chinese_embeddding_matrix=pickle.load(open("./word2vec/TextCNN_clear_without_noise_train10k_embedding_matrix.dat","rb"))
    print("加载模型")
    model=TextCNN(tokenizer.opt,chinese_embeddding_matrix).to(torch.cuda.current_device())
    model.load_state_dict(torch.load("./state_dict/TextCNN_best_model_paramters.pth"))
    token=[tokenizer.word2id[word] if word in tokenizer.word2id.keys() else 0 for word in cut_sentence(text,tokenizer.opt.stopwords_file)]
    model.eval()
    start_time = time()
    with torch.no_grad():
        tokens = torch.LongTensor(token).unsqueeze(dim=0).to(torch.cuda.current_device())
        predict_logits = model(tokens)
        labels = ['体育', '其他', '军事', '娱乐', '房产', '教育', '汽车', '游戏', '科技', '财经']
        print("消耗时间：{}——————》新闻类别：{}".format(time() - start_time, labels[torch.argmax(predict_logits, dim=-1)]))

def predict_file():
    print("加载tokenizer")
    tokenizer = pickle.load(open("./word2vec/TextCNN_clear_without_noise_train10k_without_stopwords_tokenizer_2000.dat", "rb"))
    print("加载模型")
    model=torch.load("./state_dict/TextCNN_best_model_without_stopwords_model.pth")
    data = pd.read_excel("./DataSet/newtest.xlsx")
    print("加载预测数据")
    predict_Dataset=pre_DataSet(tokenizer,data)
    predcit_DataLoader = DataLoader(predict_Dataset,batch_size=tokenizer.opt.batch_size,shuffle=False)
    model.eval()
    with torch.no_grad():
        start_time = time()
        all_labels_index=None
        print("开始预测")
        for t_batch, inputs in tqdm(enumerate(predcit_DataLoader)):
            t_outputs = model(inputs)
            y_pred = torch.argmax(t_outputs, -1)
            if all_labels_index is None:
                all_labels_index=y_pred
            else:
                all_labels_index=torch.cat((all_labels_index,y_pred),dim=0)
        labels = ['体育', '其他', '军事', '娱乐', '房产', '教育', '汽车', '游戏', '科技', '财经']
        predicted_labels=pd.DataFrame(all_labels_index)[0].apply(lambda x:labels[x])
        data.channelName=predicted_labels
        data.to_csv("pre1.csv")
        print("预测结束：{}".format(time()-start_time))

if __name__ == '__main__':
    # text = "「乒乓球」45岁“乒坛常青树”宣布退役 手握奥运入场券遗憾放弃"
    # predcit(text)
    predict_file()



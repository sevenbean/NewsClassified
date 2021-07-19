# NewsClassified(新闻文本分类)
###### 介绍一下整个项目的目录
1. NewsClassified ----------------------项目名称
   - DataSet---------------------------项目数据集
      - stopwords------------------------文本停用词
2. DataUtils-------------------------项目工具包
   - NewsDataSet.py--------------------训练集DataSet
   - pre_Dataset.py--------------------预测集DataSet
   - utils.py--------------------------项目的工具类
3. Models-----------------------------模型包
   - TextCNN.py------------------------TextCNN模型类
4. state_dict-------------------------存放训练完的模型参数
5. predict.py-------------------------模型预测
6. train.py---------------------------模型训练

---
1. 首先将新闻文本数据添加到DataSet目录中，[新闻文本训练数据集,提取码：mbtj](https://pan.baidu.com/s/1xClKnGe_bfLWkD4fxP3gyA) 
2. 在项目根目录中创建一个文件夹即word2vec
3. 将训练所需的embedding_matrix与token存在到word2vec目录中，[embedding_matrix,token下载，提取码c1rx](https://pan.baidu.com/s/1p6ckxeb1bfZ5B7-kPIg43Q)
4. 若是需要模型已经训练好的参数，需创建state_dict文件夹，并将模型参数添加进去。[模型参数,提取码2lt0](https://pan.baidu.com/s/1T4BobVByTArgUuB-4z-nbA)
5. 若是利用训练完毕的参数，则可直接运行predict.py进行预测；若是没有模型参数，则需运行train.py，再运行predict.py进行预测。[预测集,提取码：colr](https://pan.baidu.com/s/18HCSiRwckjxiiPrf6E2Dng)


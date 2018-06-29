# Chinese_NER_with_Attention

## 基于关注模型的命名实体识别算法研究
主要针对在不规范短文本中BI-LSTM-CRF模型等神经网络算法的不良性能，
加入了关注模型加以改进。所以在源代码文件夹中，仅放置了基于数据集weibo，
字级别的实验代码。其中，文件夹 BI-LSTM-CRF_char_weibo，是 BI-LSTM-CRF
模型的项目目录。文件夹 BLAC_char_weibo 是本文算法模型的项目目录。

### 实验环境：
Python3（3.6），TensorFlow1.2（及以上）。  
### 模型说明
对于一个中文句子，每个字在这个句子中有一个标签（标签列表： ｛O, B-PER,I-PER, B-LOC, I-LOC, B-ORG, I-ORG｝）。  
第一层：look-up layer，将字典中 one-hot 向量的字转换为字嵌入向量（character embedding），由于是对比实验，并未使用分布式表达来训练词嵌入向量。  
第二层：BI-LSTM layer，能够将之前和之后的特征都作为输入信息然后自动抽取出特征。  
BLAC 特殊加入层：Attention layer，将词性嵌入向量和 BI-LSTM 的隐藏层输出向量进行关注计算，得到词性对预测目标的贡献矩阵。  
第三层：CRF layer，能够为句子中的每个字标注标签。  
### 数据集说明
weibo 数据集是根据文献 [1]  中的语料修改而来。共有 15114 个命名实体。  
### 训练命令
python main.py --mode=train
### 测试命令
python main.py --mode=test --demo_model=15xxxxxxx  
其中，demo_model 参数将其设置为你在训练完成后保存模型得到的文件夹。
### 测试例句命令
python main.py --mode=demo --demo_model=15xxxxxxxx

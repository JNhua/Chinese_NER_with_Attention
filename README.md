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


### Reference
[1] 孙镇,王惠临.命名实体识别研究进展综述[J].现代图书情报技术,2010(06):42-47.  
[2] 陈基.命名实体识别综述[J].现代计算机(专业版),2016(03):24-26.  
[3] 王丹, 樊兴华. 面向短文本的命名实体识别[J]. 计算机应用, 2009, 29(1):143-145.  
[4] 刘玉娇,琚生根,李若晨,金玉.基于深度学习的中文微博命名实体识别[J].四川大学学报(工
程科学版),2016,48(S2):142-146.  
[5] Okanohara D, Miyao Y, Tsuruoka Y, et al. Improving the scalability of semi-Markov conditional
random fields for named entity recognition[C]// International Conference on Computational
Linguistics and the, Meeting of the Association for Computational Linguistics. Association for
Computational Linguistics, 2006:465-472.  
[6] Mccallum A, Li W. Early results for named entity recognition with conditional random fields,
feature induction and web-enhanced lexicons[C]// Conference on Natural Language Learning at
Hlt-Naacl. Association for Computational Linguistics, 2003:188-191.  
[7] Kazama J. Exploiting Wikipedia as external knowledge for named entity recognition[C]// Proc.
Joint Conference on Empirical Methods in Natural Language Processing and Computational
Natural Language Learning. 2007:698-707.    
[8] 隋臣. 基于深度学习的中文命名实体识别研究[D].浙江：浙江大学,2017.  
[9] Wu Y, Jiang M, Lei J, et al. Named Entity Recognition in Chinese Clinical Text Using Deep
Neural Network[J]. Stud Health Technol Inform, 2015, 216:624-628.  
[10] Huang Z, Xu W, Yu K. Bidirectional LSTM-CRF Models for Sequence Tagging[J]. arXiv
preprint, 2015, arXiv:1508.01991v1.  
[11] Chiu J P C, Nichols E. Named Entity Recognition with Bidirectional LSTM-CNNs[J]. arXiv
preprint, 2015, arXiv:1511.08308v5.  
[12] Ma X, Hovy E. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF[J]. arXiv
preprint,2016, arXiv:1603.01354v5.  
[13] Peng N, Dredze M. Named Entity Recognition for Chinese Social Media with Jointly Trained
Embeddings[C]// Conference on Empirical Methods in Natural Language Processing. 2015:548-
554.  
[14] Oliveira D M D, Laender A H F, Veloso A, et al. FS-NER:a lightweight filter-stream approach to
named entity recognition on twitter data[C]// Proceedings of the 22nd international conference
on World Wide Web companion. International World Wide Web Conferences Steering Committee,
2013:597-604.  
[15] Liu X, Zhou M. Two-stage NER for tweets with clustering[J]. Information Processing &
Management, 2013, 49(1):264-273.  
[16] 白静, 李霏, 姬东鸿. 基于注意力的BiLSTM-CNN中文微博立场检测模型[J]. 计算机应用
与软件, 2018(3):266-274.  
[17] 杨东, 王移芝. 基于 Attention-basedC-GRU 神经网络的文本分类[J]. 计算机与现代化,
2018(2).  
[18] Yang L, Ai Q, Guo J, et al. aNMM: Ranking Short Answer Texts with Attention-Based Neural
Matching Model[C]// ACM International on Conference on Information and Knowledge
Management. ACM, 2016:287-296.    
[19] Li L, Nie Y, Han W, et al. A Multi-attention-Based Bidirectional Long Short-Term Memory
Network for Relation Extraction[C]// International Conference on Neural Information Processing.
Springer, Cham, 2017:216-227.    
[20] Liu Y, Sun C, Lin L, et al. Learning Natural Language Inference using Bidirectional LSTM model
and Inner-Attention[J]. arXiv preprint,2016, arXiv:1605.09090v1.  
[21] Tan Z, Wang M, Xie J, et al. Deep Semantic Role Labeling with Self-Attention[J]. arXiv
preprint,2017, arXiv:1712.01586v1.   
[22] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[J]. arXiv preprint,2017,
arXiv:1706.03762v5.  
[23] Rei M, Crichton G K O, Pyysalo S. Attending to Characters in Neural Sequence Labeling
Models[J]. arXiv preprint,2016, arXiv:1611.04361v1.  
[24] Parikh A P, Täckström O, Das D, et al. A Decomposable Attention Model for Natural Language
Inference[J]. arXiv preprint,2016, arXiv:1606.01933v1.  
[25] Pennington J, Socher R, Manning C. Glove: Global Vectors for Word Representation[C]//
Conference on Empirical Methods in Natural Language Processing. 2014:1532-1543.  
[26] Mikolov T, Sutskever I, Chen K, et al. Distributed Representations of Words and Phrases and
their Compositionality[J]. Advances in Neural Information Processing Systems, 2013, 26:3111-
3119.  
[27] Graves A. Long Short-Term Memory[M]// Supervised Sequence Labelling with Recurrent
Neural Networks. Springer Berlin Heidelberg, 2012:1735-1780.  
[28] Mnih V, Heess N, Graves A, et al. Recurrent models of visual attention[J]. 2014, 3:2204-2212.  
[29] Bahdanau D, Cho K, Bengio Y. Neural Machine Translation by Jointly Learning to Align and
Translate[J]. arXiv preprint, 2014, arXiv:1409.0473v2.  
[30] Lample G, Ballesteros M, Subramanian S, et al. Neural Architectures for Named Entity
Recognition[J]. 2016:260-270.  

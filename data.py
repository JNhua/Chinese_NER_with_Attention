import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "B-GPE": 7, "I-GPE": 8
             }
pos2id = {
    "n": 1, "ns": 2, "nt": 3, "nr": 4,
    "ng": 5, "nrfg": 6, "nz": 7, "nrt": 8, "UNK": 0
    }

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_, pos_ = [], [], []
    for line in lines:
        if line != '\n':
            [char, label, pos] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
            pos_.append(pos)
        elif len(sent_) != 0:
            data.append((sent_, tag_, pos_))
            sent_, tag_, pos_ = [], [], []

    return data

def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_, pos_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def random_pos_embedding(pos_dim):
    pos_embedding = np.random.uniform(-0.25, 0.25, (len(pos2id), pos_dim))
    pos_embedding = np.float32(pos_embedding)
    return pos_embedding

def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, pos2id, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)
    seqs, labels, poss = [], [], []
    for (sent_, tag_, pos_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        pos = [pos2id[p] for p in pos_]
        seqs.append(sent_)
        labels.append(label_)
        poss.append(pos)
        if len(seqs) == batch_size:
            yield seqs, labels, poss
            seqs, labels, poss = [], [], []
    if len(seqs) != 0:
        yield seqs, labels, poss


def get_embedding(path):
    word2id = {}
    id = 0
    embeddings = []
    file = open(path, 'r', encoding='utf-8')
    line = file.readline()
    [word_num, embedding_dim] = line.strip().split()
    embedding_dim = int(embedding_dim)
    print("词数量应该是 "+word_num)
    lines = file.readlines()
    for line in lines:
        line_list = line.strip().split()
        word2id[line_list[0]] = id
        for l in range(1, len(line_list)):
            line_list[l] = np.float32(line_list[l])
        if len(line_list) < 101:
            diff = 101 - len(line_list)
            sum = np.sum(line_list[1:])
            mean = sum / len(line_list)
            for i in range(0, diff):
                line_list.append(mean)
        embeddings.append(line_list[1:])
        id = id + 1
    file.close()
    print("结果词数量获得的是："+str(len(word2id)))
    return embeddings, word2id, embedding_dim

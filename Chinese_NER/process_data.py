import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform


def load_data():
    train = _parse_data(open('data/train_data.data', 'rb'))
    test = _parse_data(open('data/test_data.data', 'rb'))

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

# (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = load_data()
#
# train_nump,test_nump=0,0
# train_numl,test_numl=0,0
# train_numo,test_numo=0,0
# for l in train_y:
#     for s in l:
#         if s ==1:
#             train_nump+=1
#         elif s== 3:
#             train_numl+=1
#         elif s==5:
#             train_numo+=1
# for l in test_y:
#     for s in l:
#         if s == 1:
#             test_nump += 1
#         elif s == 3:
#             test_numl += 1
#         elif s == 5:
#             test_numo += 1
#
# print("训练集",len(train_x))
# print("测试集",len(test_x))
# print("训练集人名个数",train_nump)
# print("测试集人名个数",test_nump)
# print("训练集地名个数",train_numl)
# print("测试集地名个数",test_numl)
# print("训练集机构名个数",train_numo)
# print("测试集机构名个数",test_numo)
#

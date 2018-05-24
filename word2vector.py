# -*- coding: utf-8 -*-
import os
import cPickle
import collections
try:
    with open("word_dict.pkl") as f:
        word_dict = cPickle.load(f)
except:
    print("Generating word dictionary in the first time")
    word_dict = collections.defaultdict(int)
    for dirpath, dirnames, filenames in os.walk("data"):
        if len(filenames) != 0:
            for fn in filenames:
                with open(os.path.join(dirpath, fn)) as f:
                    for line in f:
                        for w in line.strip().split():
                            word_dict[w] += 1
    items = list(word_dict.items())
    items.sort(key=lambda x: x[1], reverse=True)
    word_dict = dict()
    for i in xrange(len(items)):
        word_dict[items[i][0]] = i
    print("Saving to word_dict.pkl")
    with open("word_dict.pkl", "w") as f:
        cPickle.dump(word_dict, f, -1)
WORD_LIMIT=2000
WINDOW_SIZE=11
EMB_SIZE=32
NUM_PASSES=20
START_ID = WORD_LIMIT
END_ID = START_ID + 1
try:
    with open("all_data.pkl") as f:
        all_data = cPickle.load(f)
except:
    print("Converting words to word ids in the first time")
    all_data = []
    for dirpath, dirnames, filenames in os.walk("data"):
        for fn in filenames:
            with open(os.path.join(dirpath, fn)) as f:
                for line in f:
                    #丢弃句子中的低频词汇
                    line = [word_dict[w] for w in line.strip().split() if word_dict[w] < WORD_LIMIT]
                    line = [START_ID] + line + [END_ID]
                    if len(line) >= WINDOW_SIZE:
                        all_data.append(line)
    print("Saving to all_data.pkl")
    with open("all_data.pkl", "w") as f:
        cPickle.dump(all_data, f, -1)
import random
def word_reader_creator():
    def reader():
        global all_data
        random.shuffle(all_data)
        for line in all_data:
            for i in xrange(len(line) - WINDOW_SIZE + 1):
                yield line[i:i+WINDOW_SIZE + 1]
    return reader
import paddle.v2 as paddle
paddle.init(use_gpu=False, trainer_count=3)
words = [paddle.layer.data(name="word_%d" % i, type=paddle.data_type.integer_value(WORD_LIMIT + 2)) for i in xrange(WINDOW_SIZE)]



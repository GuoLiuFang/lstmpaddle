# coding: utf-8
import collections
import tarfile
import re
import string
import jieba
__all__ = ['word_dict', 'train', 'test']
def word_dict(cutoff=150):
    """
    从语料中建立一个. Build a word dictionary from the corpus.
    :return: Word dictionary
    :rtype: dict
    """
    return build_dict(re.compile("glodon/((train)|(test))/key./.*?/.*\.txt$"), cutoff)
def build_dict(pattern, cutoff):
    """
    Build a word dictionary from the corpus. Keys of 
    the dictionary are words, and values are zero-based
    IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    for doc, _ in tokenize(pattern):
        for word in doc:
            word_freq[word] += 1
    # not sure if i should prune less-frequent words here.
    word_freq = filter(lambda x: x[1] > cutoff, word_freq.items())
    
    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words)
    return word_idx
def tokenize(pattern):
    """
    Read files that match the given pattern.
    Tokenize and yield each file.
    :return:(list,label)
    """
    labelpattern = re.compile("glodon/((train)|(test))/key./(?P<label>.*?)/.*\.txt$")
    with tarfile.open('glodon.tar.gz') as tarf:
        tf = tarf.next()
        while tf != None:
            if bool(pattern.match(tf.name)):
                # newline and punctuations removal and ad-hoc tokenization.
#                 print("filename is %s" % tf.name)
                match = labelpattern.match(tf.name)
                label_idx = match.group("label")
#                 print('<>'.join(jieba.cut(tarf.extractfile(tf).read().rstrip("\n\r").translate(None, string.punctuation).lower())))
                yield (jieba.cut(tarf.extractfile(tf).read().rstrip("\n\r").translate(None, string.punctuation).lower()), label_idx)
            tf = tarf.next()

# x = word_dict()
# for y in sorted(x, key=x.get, reverse=True):
#     print("key---%s----value--%s" % (y,x[y]))

def train(word_idx, level=3):
    """
    training set creator..
    It returns a reader creator , each sample
    in the reader is an zero-based ID sequence 
    and label in [0,1]
    """
    return reader_creator(re.compile("glodon/train/key%d/.*?/.*\.txt$" % level), word_idx)
def reader_creator(pattern, word_idx):
    UNK = word_idx['<unk>']
    INS = []
    def load(pattern, out):
        for doc, label in tokenize(pattern):
            out.append(([word_idx.get(w, UNK) for w in doc], label))
    load(pattern, INS)
    def reader():
        for doc, label in INS:
            yield doc, int(label)
    return reader
# #         def reader():
#     for doc, label in INS:
#         yield doc, label
# #     return reader
def test(word_idx, level=3):
    return reader_creator(re.compile("glodon/test/key%d/.*?/.*\.txt$" % level), word_idx)

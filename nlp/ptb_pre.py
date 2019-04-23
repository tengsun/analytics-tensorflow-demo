import codecs
import collections
from operator import itemgetter


RAW_DATA = './data/simple-examples/data/ptb.train.txt'
VOCAB = './data/ptb.vocab'
OUTPUT_DATA = './data/ptb.train'


def generate_vocab():
    # count words
    counter = collections.Counter()
    with codecs.open(RAW_DATA, 'r', 'utf-8') as fi:
        for line in fi:
            for word in line.strip().split():
                counter[word] += 1
    
    # sort words by count
    sorted_words_to_count = sorted(
        counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_words_to_count]
    sorted_words.append('<eos>')

    # write words into vocab
    with codecs.open(VOCAB, 'w', 'utf-8') as fo:
        for word in sorted_words:
            fo.write(word + '\n')


def word_to_id():
    # convert word to id
    with codecs.open(VOCAB, 'r', 'utf-8') as fv:
        vocab = [w.strip() for w in fv.readlines()]
    word2id = {k:v for (k,v) in zip(vocab, range(len(vocab)))}
    return word2id


def get_word_id(word, word2id):
    return word2id[word] if word in word2id else word2id['<unk>']


def replace_by_id():
    fi = codecs.open(RAW_DATA, 'r', 'utf-8')
    fo = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
    word2id = word_to_id()

    for line in fi:
        words = line.strip().split() + ['<eos>']
        output_line = ' '.join([str(get_word_id(w, word2id)) for w in words]) + '\n'
        fo.write(output_line)
    
    fi.close()
    fo.close()


if __name__ == '__main__':
    generate_vocab()
    # replace_by_id()

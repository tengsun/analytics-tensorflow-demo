import codecs
import collections
from operator import itemgetter


RAW_DATA = './data/simple-examples/data/ptb.train.txt'
VOCAB_OUTPUT = './data/ptb.vocab'


def word_sort():
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
    # sorted_words.append('<eos>')

    # write words into vocab
    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as fo:
        for word in sorted_words:
            fo.write(word + '\n')


if __name__ == '__main__':
    word_sort()

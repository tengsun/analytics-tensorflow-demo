import tensorflow as tf


MAX_LEN = 50
SOS_ID = 0


# read line from dataset file
def make_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # cut words and store in 1-dim array
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # convert words string into int value
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # count words size and save into dataset
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def make_src_trg_dataset(src_path, trg_path, batch_size):
    # read source and target data
    src_data = make_dataset(src_path)
    trg_data = make_dataset(trg_path)

    # combine src and trg dataset
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # remove empty and lengthy sentences
    def filter_length(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(filter_length)

    # decoder input format: <sos> X Y Z
    # decoder label format: X Y Z <eos>
    def make_trg_input(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(make_trg_input)

    # shuffle the training data
    dataset = dataset.shuffle(10000)

    padded_shapes = (
        # encoder input unknown, length is a digit
        (tf.TensorShape([None]), tf.TensorShape([])),
        # decoder input unknown, output unknown, length is a digit
        (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset

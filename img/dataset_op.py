import tensorflow as tf


def tfrecord_parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        })
    return features


def basic_op():
    # array data
    # input_data = [1, 2, 3, 5, 8]
    # dataset = tf.data.Dataset.from_tensor_slices(input_data)

    # file data
    # input_data = ['./data/text_file1.txt', './data/text_file2.txt']
    # dataset = tf.data.TextLineDataset(input_data)

    # tfrecord data
    input_data = ['./data/tfrecords-000-of-003', './data/tfrecords-001-of-003', 
        './data/tfrecords-002-of-003']
    dataset = tf.data.TFRecordDataset(input_data)
    dataset = dataset.map(tfrecord_parser)

    # get iterator from dataset
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()

    with tf.Session() as sess:
        for i in range(len(input_data)):
            print(sess.run(x))


if __name__ == '__main__':
    basic_op()

import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tf_record():
    num_shards = 3
    instances_per_shard = 10
    for i in range(num_shards):
        filename = ('./data/tfrecords-%.3d-of-%.3d' %(i, num_shards))
        writer = tf.python_io.TFRecordWriter(filename)

        # write image as tfrecord
        for j in range(instances_per_shard):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature('img_' + str(i) + '_' + str(j)),
                    'height': _int64_feature(j * 2),
                    'width': _int64_feature(j * 2)
                }
            ))
            writer.write(example.SerializeToString())
        writer.close()


def inject_tf_record():
    # load files and init queue
    files = tf.train.match_filenames_once('./data/tfrecords-*')
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    # read tfrecords
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        })
    image, height, width = features['image'], features['height'], features['width']

    # define batch size and capacity
    img_batch, h_batch, w_batch = tf.train.shuffle_batch([image, height, width], 
        batch_size=5, capacity=100, min_after_dequeue=10)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()

        # start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # print 3 values in queue
        for _ in range(6):
            curr_img_batch, curr_h_batch, curr_w_batch = sess.run(
                [img_batch, h_batch, w_batch])
            print(curr_img_batch, curr_h_batch, curr_w_batch)
        
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # write_tf_record()
    inject_tf_record()

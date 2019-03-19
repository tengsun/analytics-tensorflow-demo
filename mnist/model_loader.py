import tensorflow as tf

def main(argv=None):
    saver = tf.train.import_meta_graph('./model/model_saver.ckpt.meta')

    with tf.Session() as sess:
        saver.restore(sess, './model/model_saver.ckpt')
        print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))

if __name__ == '__main__':
    tf.app.run()

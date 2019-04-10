import tensorflow as tf


def cal_cross_entropy():
    # vocabulary length is 3, here are two words
    word_labels = tf.constant([2, 0])
    # logits is not probability, not in 0.0~1.0
    predict_logits = tf.constant([[2.0, -1.0, 3.0], [1.0, 0.0, -0.5]])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=word_labels, logits=predict_logits)
    sess = tf.Session()
    print(sess.run(loss))

    # word probability distribution
    word_prob_dist = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=word_prob_dist, logits=predict_logits)
    print(sess.run(loss))

    # smooth data to avoid overfitting
    word_prob_smooth = tf.constant([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]])
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=word_prob_smooth, logits=predict_logits)
    print(sess.run(loss))


if __name__ == '__main__':
    cal_cross_entropy()

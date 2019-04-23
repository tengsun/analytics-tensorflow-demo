import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


CKPT_PATH = '../model/seq2seq_ckpt-100'

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 770
TRG_VOCAB_SIZE = 836
SHARE_EMB_AND_SOFTMAX = True

SOS_ID = 0
EOS_ID = 1


class NMTModel(object):
    def __init__(self):
        # init encode and decode cells
        self.encode_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])
        self.decode_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])

        # source and target word vector
        self.src_embedding = tf.get_variable(
            'src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            'trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # define variables in softmax layer
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                'softmax_weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            'softmax_bias', [TRG_VOCAB_SIZE])
    
    def inference(self, src_input):
        # convert sentence to batch (size=1)
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # use dynamic rnn to build encoder
        with tf.variable_scope('encoder'):
            encode_outputs, encode_state = tf.nn.dynamic_rnn(
                self.encode_cell, src_emb, src_size, dtype=tf.float32)
        
        # set maximum decoder length
        MAX_DEC_LEN = 100
        with tf.variable_scope('decoder/rnn/multi_rnn_cell'):
            # use dynamic array to save generated sentence
            init_array = tf.TensorArray(dtype=tf.int32, size=0, 
                dynamic_size=True, clear_after_read=False)
            # set 1st word as <sos>
            init_array = init_array.write(0, SOS_ID)
            init_loop_var = (encode_state, init_array, 0)

            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID), 
                    tf.less(step, MAX_DEC_LEN-1)))
            
            def loop_body(state, trg_ids, step):
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
                decode_outputs, next_state = self.decode_cell.call(
                    state=state, inputs=trg_emb)
                output = tf.reshape(decode_outputs, [-1, HIDDEN_SIZE])
                logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1
            
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def key_in_checkpoint():
    reader = pywrap_tensorflow.NewCheckpointReader(CKPT_PATH)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


def main():
    with tf.variable_scope('nmt_model', reuse=None):
        model = NMTModel()

    # define a test case
    test_sentence = [34, 11, 10, 127, 3]
    output_op = model.inference(test_sentence)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CKPT_PATH)
    output = sess.run(output_op)
    sess.close()
    print(output)


if __name__ == '__main__':
    main()

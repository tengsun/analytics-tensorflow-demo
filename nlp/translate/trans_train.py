import tensorflow as tf
import trans_batch


SRC_TRAIN_DATA = '../data/nmt.train.en'
TRG_TRAIN_DATA = '../data/nmt.train.zh'
CHECKPOINT_PATH = '../model/seq2seq_ckpt'
SRC_VOCAB_SIZE = 770
TRG_VOCAB_SIZE = 836

# LSTM network params
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
BATCH_SIZE = 5
NUM_EPOCH = 3
DROPOUT_KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True


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
    
    # define the forward computing graph
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        # convert input/output word into embedding
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # apply dropout on word embedding
        src_emb = tf.nn.dropout(src_emb, DROPOUT_KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, DROPOUT_KEEP_PROB)

        # use dynamic rnn to build encoder
        with tf.variable_scope('encoder'):
            encode_outputs, encode_state = tf.nn.dynamic_rnn(
                self.encode_cell, src_emb, src_size, dtype=tf.float32)
        
        # use encode state to init decoder
        with tf.variable_scope('decoder'):
            decode_outputs, decode_state = tf.nn.dynamic_rnn(
                self.decode_cell, trg_emb, trg_size, initial_state=encode_state)
        
        # calculate log perplexity as loss
        output = tf.reshape(decode_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)

        # set padding places' weight to 0
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # control the step size of gradients
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        
        return cost_per_token, train_op


# run epoch and save checkpoint every 200 steps
def run_epoch(session, cost_op, train_op, saver, step):
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print('after {0} steps, per token cost is {1:.3f}'.format(step, cost))
            if step % 100 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    # define the init function
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # define the rnn train model
    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel()
    
    # define the model input data
    data = trans_batch.make_src_trg_dataset(SRC_TRAIN_DATA, 
        TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # define the forward propagation
    cost_op, train_op = train_model.forward(src, src_size, 
        trg_input, trg_label, trg_size)
    
    # train the rnn model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print('in epoch: {0}'.format(i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
    main()

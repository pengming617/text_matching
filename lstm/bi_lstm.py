import tensorflow as tf
import math


class Bi_lstm(object):

    def __init__(self, is_trainning, seq_length, class_num, vocabulary_size, embedding_size, hidden_num,
                l2_lambda, learning_rate):
        self.is_trainning = is_trainning
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_num = hidden_num
        self.seq_length = seq_length
        self.class_num = class_num

        # init placeholder
        self.text_a = tf.placeholder(tf.int32, [None, seq_length], name='text_a')
        self.text_b = tf.placeholder(tf.int32, [None, seq_length], name='text_b')
        self.y = tf.placeholder(tf.int32, [None, class_num], name='y')
        # real length
        self.a_length = tf.placeholder(tf.int32, [None], name='a_length')
        self.b_length = tf.placeholder(tf.int32, [None], name='b_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding层 论文中采用是预训练好的词向量 这里随机初始化一个词典 在训练过程中进行调整
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.vocab_matrix = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_size],
                                                                stddev=1.0 / math.sqrt(embedding_size)),
                                            name='vacab_matrix')
            self.text_a_embed = tf.nn.embedding_lookup(self.vocab_matrix, self.text_a)
            self.text_b_embed = tf.nn.embedding_lookup(self.vocab_matrix, self.text_b)

        with tf.name_scope('Input_Encoding'):
            a = self.biLSTMBlock(self.text_a_embed, hidden_num, 'Input_Encoding/biLSTM', self.a_length)
            b = self.biLSTMBlock(self.text_b_embed, hidden_num, 'Input_Encoding/biLSTM', self.b_length, isreuse=True)

        diff = tf.subtract(a, b)

        with tf.name_scope("output"):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('feed_foward'):
                outputs = tf.nn.dropout(diff, self.dropout_keep_prob)
                outputs = tf.reshape(outputs, [-1, hidden_num * seq_length * 2])
                outputs = tf.layers.dense(outputs, hidden_num, tf.nn.relu, kernel_initializer=initializer)
                self.logits = tf.layers.dense(outputs, class_num, tf.nn.tanh, kernel_initializer=initializer)
            self.score = tf.nn.softmax(self.logits, name='score')
            self.prediction = tf.argmax(self.score, 1, name="prediction")

        with tf.name_scope('cost'):
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            self.cost = tf.reduce_mean(self.cost)
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            self.loss = l2_loss + self.cost

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32))

        if not is_trainning:
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def biLSTMBlock(self, inputs, num_units, scope, seq_len=None, isreuse=False):
        with tf.variable_scope(scope, reuse=isreuse):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)

            (a_outputs, a_output_states) = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell,
                                                                           inputs,
                                                                           sequence_length=seq_len,
                                                                           dtype=tf.float32)
            a_bar = tf.concat(a_outputs, axis=2)
            return a_bar


if __name__ == '__main__':
    esim = Bi_lstm(True, 20, 2, 10000, 300, 300, 0.001, 0.0001)
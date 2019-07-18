import tensorflow as tf
import numpy as np
import math

'''
Each config should be a dictionary with entries:
type	:	Type pf convolutional layer depending of use of attenion map; one of BCNN, ABCNN-1, ABCNN-2, ABCNN-3
w		:	Width of convolutional kernel
n		:	Number of convolutional ops
nl		:	Type of non-linearity; one of 'tanh' or 'relu'
'''
DEFAULT_CONFIG = [{'type': 'ABCNN-1', 'w': 3, 'n': 50, 'nl': 'tanh'} for _ in range(3)]


class ABCNN:

    def __init__(self, is_trainning, conv_layers, embed_size, vocabulary_size, sentence_len, external_measures=0, config=DEFAULT_CONFIG):
        self.conv_layers = conv_layers
        self.embed_size = embed_size
        self.vocabulary_size = vocabulary_size
        self.sentence_len = sentence_len
        self.external_measures = external_measures
        self.config = config
        self.is_trainning = is_trainning

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.name_scope("data"):
            self.text_a = tf.placeholder(tf.int32, shape=[None, self.sentence_len], name='text_a')
            self.text_b = tf.placeholder(tf.int32, shape=[None, self.sentence_len], name='text_b')
            if self.external_measures > 0:
                self.ext = tf.placeholder(tf.float32, shape=[None, self.external_measures], name='ext_measure')
            self.y = tf.placeholder(tf.int32, shape=[None, 2], name="y")

        # embedding层 论文中采用是预训练好的词向量 这里随机初始化一个词典 在训练过程中进行调整
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.vocab_matrix = tf.Variable(tf.truncated_normal(shape=[self.vocabulary_size, self.embed_size],
                                                                stddev=1.0 / math.sqrt(self.embed_size)),
                                            name='vacab_matrix')
            self.x1 = tf.nn.embedding_lookup(self.vocab_matrix, self.text_a)
            self.x2 = tf.nn.embedding_lookup(self.vocab_matrix, self.text_b)
            tq1 = tf.transpose(self.x1, [0, 2, 1], name='t_q1')
            tq2 = tf.transpose(self.x2, [0, 2, 1], name='t_q2')
            self.q1 = tf.expand_dims(tq1, -1, name='q1')
            self.q2 = tf.expand_dims(tq2, -1, name='q2')

        layer_input = [self.q1, self.q2]
        for i in range(self.conv_layers):
            last = (i == self.conv_layers - 1)
            if self.config[i]['type'] == 'ABCNN-3':
                layer_input = self._add_ABCNN_3(i, self.config[i], last, layer_input)
                continue
            if self.config[i]['type'] == 'ABCNN-2':
                layer_input = self._add_ABCNN_2(i, self.config[i], last, layer_input)
                continue
            if self.config[i]['type'] == 'ABCNN-1':
                layer_input = self._add_ABCNN_1(i, self.config[i], last, layer_input)
                continue
            if self.config[i]['type'] == 'BCNN':
                layer_input = self._add_BCNN(i, self.config[i], last, layer_input)
                continue
            else:
                raise ValueError('Unrecognised conv layer type')

        with tf.variable_scope('fc') as scope:
            if self.external_measures > 0:
                fc_in = tf.concat([layer_input[0], layer_input[1], self.ext], axis=1)
            else:
                fc_in = tf.concat([layer_input[0], layer_input[1]], axis=1)
            w = tf.Variable(tf.truncated_normal([fc_in.get_shape().as_list()[1], 2], stddev=0.1, dtype=tf.float32),
                            name='weights')
            b = tf.Variable(tf.zeros([2], dtype=tf.float32), name="bias")

            logits = tf.matmul(fc_in, w) + b

        with tf.name_scope('loss'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits)
            self.cross_entropy = tf.reduce_mean(self.cross_entropy)
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * 0.01
            self.loss = l2_loss + self.cross_entropy

        self.score = tf.nn.softmax(logits, name='score')
        self.prediction = tf.argmax(self.score, 1, name="prediction")

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32))

        if not self.is_trainning:
            return

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_step')

    def _conv_layer(self, config, input):
        kernel = tf.get_variable('kernel', [input.get_shape()[1], config['w'], input.get_shape()[3], config['n']],
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='VALID')
        biases = tf.get_variable("biases", config['n'], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        if config['nl'] == 'tanh':
            nl = tf.nn.tanh
        elif config['nl'] == 'relu':
            nl = tf.nn.relu
        else:
            raise ValueError('_conv_layer: %s is not implemented' % config['nl'])
        return nl(conv + biases)

    def _add_BCNN(self, id, config, last, layer_input):
        scope_name = 'BCNN_' + str(id)
        with tf.variable_scope(scope_name, initializer=tf.contrib.layers.xavier_initializer()) as scope:
            with tf.variable_scope('conv') as scope:
                padded_in1 = tf.pad(layer_input[0], [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]],
                                    'constant')
                padded_in2 = tf.pad(layer_input[1], [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]],
                                    'constant')
                conv1 = self._conv_layer(config, padded_in1)
                scope.reuse_variables()
                conv2 = self._conv_layer(config, padded_in2)

            if last:
                with tf.variable_scope('all-ap') as scope:
                    ap1 = tf.reduce_mean(conv1, axis=[1, 2])
                    ap2 = tf.reduce_mean(conv2, axis=[1, 2])
                    return ap1, ap2
            else:
                with tf.variable_scope('%d-ap' % config['w']) as scope:
                    avg_pool1 = tf.nn.avg_pool(conv1, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool1')
                    avg_pool2 = tf.nn.avg_pool(conv2, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool2')
                    ap1 = tf.transpose(avg_pool1, perm=[0, 3, 2, 1], name='ap1')
                    ap2 = tf.transpose(avg_pool2, perm=[0, 3, 2, 1], name='ap2')
                    return ap1, ap2

    def _add_ABCNN_1(self, id, config, last, layer_input):
        scope_name = 'ABCNN1_' + str(id)
        with tf.variable_scope(scope_name, initializer=tf.contrib.layers.xavier_initializer()) as scope:
            with tf.variable_scope('similarity') as scope:
                tile_in1 = tf.tile(layer_input[0], [1, 1, 1, layer_input[1].get_shape().as_list()[2]], name='tile1')
                tile_in2 = tf.transpose(layer_input[1], [0, 1, 3, 2], name='tile2')
                sq_dist = tf.squared_difference(tile_in1, tile_in2, name='sq_dist')
                pair_dist = tf.sqrt(tf.reduce_sum(sq_dist, axis=[1]), name='pair_dist')
                similarity = tf.reciprocal(tf.add(pair_dist, tf.constant(1.0, dtype=tf.float32, name='one')),
                                           name='similarity')

            with tf.variable_scope('attention') as scope:
                W = tf.get_variable('W', [layer_input[0].get_shape()[2], layer_input[0].get_shape()[1]],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                A1 = tf.matrix_transpose(tf.einsum("ijk,kl->ijl", similarity, W))
                A2 = tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(similarity), W))

            with tf.variable_scope('conv') as scope:
                layer_in1 = tf.concat([layer_input[0], tf.expand_dims(A1, -1)], axis=3)
                layer_in2 = tf.concat([layer_input[1], tf.expand_dims(A2, -1)], axis=3)
                padded_in1 = tf.pad(layer_in1, [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]], 'constant')
                padded_in2 = tf.pad(layer_in2, [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]], 'constant')
                conv1 = self._conv_layer(config, padded_in1)
                scope.reuse_variables()
                conv2 = self._conv_layer(config, padded_in2)

            if last:
                with tf.variable_scope('all-ap') as scope:
                    ap1 = tf.reduce_mean(conv1, axis=[1, 2])
                    ap2 = tf.reduce_mean(conv2, axis=[1, 2])
                    return ap1, ap2
            else:
                with tf.variable_scope('%d-ap' % config['w']) as scope:
                    avg_pool1 = tf.nn.avg_pool(conv1, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool1')
                    avg_pool2 = tf.nn.avg_pool(conv2, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool2')
                    ap1 = tf.transpose(avg_pool1, perm=[0, 3, 2, 1], name='ap1')
                    ap2 = tf.transpose(avg_pool2, perm=[0, 3, 2, 1], name='ap2')
                    return ap1, ap2

    def _add_ABCNN_2(self, id, config, last, layer_input):
        scope_name = 'ABCNN2_' + str(id)
        with tf.variable_scope(scope_name, initializer=tf.contrib.layers.xavier_initializer()) as scope:
            with tf.variable_scope('conv') as scope:
                padded_in1 = tf.pad(layer_input[0], [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]],
                                    'constant')
                padded_in2 = tf.pad(layer_input[1], [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]],
                                    'constant')
                conv1 = self._conv_layer(config, padded_in1)
                scope.reuse_variables()
                conv2 = self._conv_layer(config, padded_in2)

            with tf.variable_scope('similarity') as scope:
                conv1t = tf.transpose(conv1, [0, 3, 2, 1], name='conv1t')
                conv2t = tf.transpose(conv2, [0, 3, 2, 1], name='conv2t')
                tile_out1 = tf.tile(conv1t, [1, 1, 1, conv2.get_shape().as_list()[2]], name='tile1')
                tile_out2 = tf.transpose(conv2t, [0, 1, 3, 2], name='tile2')
                sq_dist = tf.squared_difference(tile_out1, tile_out2, name='sq_dist')
                pair_dist = tf.sqrt(tf.reduce_sum(sq_dist, axis=[1]), name='pair_dist')
                similarity = tf.reciprocal(tf.add(pair_dist, tf.constant(1.0, dtype=tf.float32, name='one')),
                                           name='similarity')

            with tf.variable_scope('attention') as scope:
                A1 = tf.reduce_sum(similarity, axis=[2], name='attention_map1')
                A2 = tf.reduce_sum(similarity, axis=[1], name='attention_map2')
                A1e = tf.expand_dims(A1, 1)
                A2e = tf.expand_dims(A2, 1)
                A1f = tf.expand_dims(A1e, -1)
                A2f = tf.expand_dims(A2e, -1)
                conv1w = tf.multiply(conv1t, A1f, name='weighted_conv1')
                conv2w = tf.multiply(conv2t, A2f, name='weighted_conv2')

            if last:
                with tf.variable_scope('all-ap') as scope:
                    ap1 = tf.reduce_mean(conv1w, axis=[2, 3])
                    ap2 = tf.reduce_mean(conv2w, axis=[2, 3])
                    return ap1, ap2
            else:
                with tf.variable_scope('%d-ap' % config['w']) as scope:
                    avg_pool1 = tf.nn.avg_pool(conv1w, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool1')
                    avg_pool2 = tf.nn.avg_pool(conv2w, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool2')
                    return avg_pool1, avg_pool2

    def _add_ABCNN_3(self, id, config, last, layer_input):
        scope_name = 'ABCNN3_' + str(id)
        with tf.variable_scope(scope_name, initializer=tf.contrib.layers.xavier_initializer()) as scope:
            with tf.variable_scope('similarity_pre') as scope:
                tile_in1 = tf.tile(layer_input[0], [1, 1, 1, layer_input[1].get_shape().as_list()[2]], name='tile1')
                tile_in2 = tf.transpose(layer_input[1], [0, 1, 3, 2], name='tile2')
                sq_dist = tf.squared_difference(tile_in1, tile_in2, name='sq_dist')
                pair_dist = tf.sqrt(tf.reduce_sum(sq_dist, axis=[1]), name='pair_dist')
                similarity = tf.reciprocal(tf.add(pair_dist, tf.constant(1.0, dtype=tf.float32, name='one')),
                                           name='similarity')

            with tf.variable_scope('attention_pre') as scope:
                W = tf.get_variable('W', [layer_input[0].get_shape()[1], layer_input[0].get_shape()[2]],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                A1 = tf.matrix_transpose(tf.einsum("ijk,kl->ijl", similarity, W))
                A2 = tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(similarity), W))

            with tf.variable_scope('conv') as scope:
                layer_in1 = tf.concat([layer_input[0], tf.expand_dims(A1, -1)], axis=3)
                layer_in2 = tf.concat([layer_input[1], tf.expand_dims(A2, -1)], axis=3)
                padded_in1 = tf.pad(layer_in1, [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]], 'constant')
                padded_in2 = tf.pad(layer_in2, [[0, 0], [0, 0], [config['w'] - 1, config['w'] - 1], [0, 0]], 'constant')
                conv1 = self._conv_layer(config, padded_in1)
                scope.reuse_variables()
                conv2 = self._conv_layer(config, padded_in2)

            with tf.variable_scope('similarity_post') as scope:
                conv1t = tf.transpose(conv1, [0, 3, 2, 1], name='conv1t')
                conv2t = tf.transpose(conv2, [0, 3, 2, 1], name='conv2t')
                tile_out1 = tf.tile(conv1t, [1, 1, 1, conv2.get_shape().as_list()[2]], name='tile1')
                tile_out2 = tf.transpose(conv2t, [0, 1, 3, 2], name='tile2')
                sq_dist = tf.squared_difference(tile_out1, tile_out2, name='sq_dist')
                pair_dist = tf.sqrt(tf.reduce_sum(sq_dist, axis=[1]), name='pair_dist')
                similarity = tf.reciprocal(tf.add(pair_dist, tf.constant(1.0, dtype=tf.float32, name='one')),
                                           name='similarity')

            with tf.variable_scope('attention_post') as scope:
                A1 = tf.reduce_sum(similarity, axis=[2], name='attention_map1')
                A2 = tf.reduce_sum(similarity, axis=[1], name='attention_map2')
                A1e = tf.expand_dims(A1, 1)
                A2e = tf.expand_dims(A2, 1)
                A1f = tf.expand_dims(A1e, -1)
                A2f = tf.expand_dims(A2e, -1)
                conv1w = tf.multiply(conv1t, A1f, name='weighted_conv1')
                conv2w = tf.multiply(conv2t, A2f, name='weighted_conv2')

            if last:
                with tf.variable_scope('all-ap') as scope:
                    ap1 = tf.reduce_mean(conv1w, axis=[2, 3])
                    ap2 = tf.reduce_mean(conv2w, axis=[2, 3])
                    return ap1, ap2
            else:
                with tf.variable_scope('%d-ap' % config['w']) as scope:
                    avg_pool1 = tf.nn.avg_pool(conv1w, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool1')
                    avg_pool2 = tf.nn.avg_pool(conv2w, [1, 1, config['w'], 1], strides=[1, 1, 1, 1], padding="VALID",
                                               name='avg_pool2')
                    return avg_pool1, avg_pool2


if __name__ == '__main__':
    c = ABCNN(True, conv_layers=3, embed_size=50, vocabulary_size=1000, sentence_len=40)

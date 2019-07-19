import tensorflow as tf
import numpy as np
import math


class ABCNN():
    def __init__(self, is_trainning, s, w, l2_reg, model_type, vocabulary_size, d0=300, di=50,
                 num_classes=2, num_layers=2):
        """
        Implmenentaion of ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)

        :param s: sentence length
        :param w: filter width
        :param l2_reg: L2 regularization coefficient
        :param model_type: Type of the network(BCNN, ABCNN1, ABCNN2, ABCNN3).
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_classes: The number of classes for answers.
        :param num_layers: The number of convolution layers.
        """
        self.is_trainning = is_trainning
        self.text_a = tf.placeholder(tf.int32, shape=[None, s], name="text_a")
        self.text_b = tf.placeholder(tf.int32, shape=[None, s], name="text_b")
        self.y = tf.placeholder(tf.int32, shape=[None, num_classes], name="y")
        # self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")

        # embedding层 论文中采用是预训练好的词向量 这里随机初始化一个词典 在训练过程中进行调整
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.vocab_matrix = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, d0],
                                                                stddev=1.0 / math.sqrt(d0)),
                                            name='vacab_matrix')
            self.x1 = tf.nn.embedding_lookup(self.vocab_matrix, self.text_a)
            self.x2 = tf.nn.embedding_lookup(self.vocab_matrix, self.text_b)

        self.x1 = tf.transpose(self.x1, [0, 2, 1])
        self.x2 = tf.transpose(self.x2, [0, 2, 1])

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
            return 1 / (1 + euclidean)

        def make_attention_mat(x1, x2):
            # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
            # x2 => [batch, height, 1, width]
            # [batch, width, wdith] = [batch, s, s]

            # 作者论文中提出计算attention的方法 在实际过程中反向传播计算梯度时 容易出现NaN的情况 这里面加以修改
            # euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            # return 1 / (1 + euclidean)

            x1 = tf.transpose(tf.squeeze(x1, [-1]), [0, 2, 1])
            attention = tf.einsum("ijk,ikl->ijl", x1, tf.squeeze(x2, [-1]))
            return attention

        def convolution(name_scope, x, d, reuse):
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope("conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=di,
                        kernel_size=(d, w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=reuse,
                        trainable=True,
                        scope=scope
                    )
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                    # [batch, di, s+w-1, 1]
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    return conv_trans

        def w_pool(variable_scope, x, attention):
            # x: [batch, di, s+w-1, 1]
            # attention: [batch, s+w-1]
            with tf.variable_scope(variable_scope + "-w_pool"):
                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    pools = []
                    # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                    attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

                    for i in range(s):
                        # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                        pools.append(tf.reduce_sum(x[:, :, i:i + w, :] * attention[:, :, i:i + w, :],
                                                   axis=2,
                                                   keep_dims=True))

                    # [batch, di, s, 1]
                    w_ap = tf.concat(pools, axis=2, name="w_ap")
                else:
                    w_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, w),
                        strides=1,
                        padding="VALID",
                        name="w_ap"
                    )
                    # [batch, di, s, 1]

                return w_ap

        def all_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width = s
                    d = d0
                else:
                    pool_width = s + w - 1
                    d = di

                all_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=(1, pool_width),
                    strides=1,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]

                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])

                return all_ap_reshaped

        def CNN_layer(variable_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                if model_type == "ABCNN1" or model_type == "ABCNN3":
                    with tf.name_scope("att_mat"):
                        aW = tf.get_variable(name="aW",
                                             shape=(s, d),
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))

                        # [batch, s, s]
                        att_mat = make_attention_mat(x1, x2)
                        # att_mat = tf.get_variable('att_mat', [None, x1.get_shape()[2], x1.get_shape()[2]],
                        #             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

                        # [batch, s, s] * [s,d] => [batch, s, d]
                        # matrix transpose => [batch, d, s]
                        # expand dims => [batch, d, s, 1]
                        x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                        x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat),
                                                                            aW)), -1)

                        # [batch, d, s, 2]
                        x1 = tf.concat([x1, x1_a], axis=3)
                        x2 = tf.concat([x2, x2_a], axis=3)

                left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=False)
                right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=True)

                left_attention, right_attention = None, None

                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    # [batch, s+w-1, s+w-1]
                    att_mat = make_attention_mat(left_conv, right_conv)
                    # [batch, s+w-1], [batch, s+w-1]
                    left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

                left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention)
                left_ap = all_pool(variable_scope="left", x=left_conv)
                right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention)
                right_ap = all_pool(variable_scope="right", x=right_conv)

                return left_wp, left_ap, right_wp, right_ap

        x1_expanded = tf.expand_dims(self.x1, -1)
        x2_expanded = tf.expand_dims(self.x2, -1)

        self.LO_0 = all_pool(variable_scope="input-left", x=x1_expanded)
        self.RO_0 = all_pool(variable_scope="input-right", x=x2_expanded)

        LI_1, self.LO_1, RI_1, self.RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        self.sims = [cos_sim(self.LO_0, self.RO_0), cos_sim(self.LO_1, self.RO_1)]

        if num_layers > 1:
            _, LO_2, _, RO_2 = CNN_layer(variable_scope="CNN-2", x1=LI_1, x2=RI_1, d=di)
            self.test = LO_2
            self.test2 = RO_2
            self.sims.append(cos_sim(LO_2, RO_2))

        with tf.variable_scope("output-layer"):
            # self.output_features = tf.concat([self.features, tf.stack(sims, axis=1)], axis=1, name="output_features")
            self.output_features = tf.stack(self.sims, axis=1)

            self.logits = tf.contrib.layers.fully_connected(
                inputs=self.output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC")

        self.score = tf.nn.softmax(self.logits, name='score')
        self.prediction = tf.argmax(self.score, 1, name="prediction")
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), self.prediction), tf.float32))

        with tf.name_scope('cost'):
            self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
            self.cost = tf.reduce_mean(self.cost)
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * 0.01
            self.loss = l2_loss + self.cost

        if not is_trainning:
            return

        tvars = tf.trainable_variables()

        self.gradients = tf.gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(self.gradients, 10)

        optimizer = tf.train.AdamOptimizer(0.001)
        # grad_check = tf.check_numerics(grads, 'check_numerics caught bad gradients')
        # with tf.control_dependencies([grad_check]):
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    abcnn = ABCNN(True, 20, 3, 0.001, 'ABCNN1', vocabulary_size=1000, d0=300, di=50, num_classes=2, num_layers=1)

import tensorflow as tf
import numpy as np


class DSSM(object):

    def __init__(self, unigram_num, maxsim_num, unsim_num, hidden_layer, is_normalize, is_trainning):

        self.query = tf.placeholder(tf.int32, [None, unigram_num], name="query")
        self.sim_text = tf.placeholder(tf.int32, [None, maxsim_num, unigram_num], name='sim_text')
        self.unsim_text = tf.placeholder(tf.int32, [None, unsim_num, unigram_num], name='unsim_text')
        self.sim_text_num = tf.placeholder(tf.int32, [None], name='sim_text_num')

        # FC层
        with tf.name_scope("fc_layer"):
            L1_N = unigram_num
            for hidden_num in hidden_layer:
                L2_N = hidden_num
                l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
                weights = tf.Variable(tf.random_uniform([L1_N, L2_N], minval=-l2_par_range, maxval=l2_par_range))
                bias = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))

                self.query = tf.nn.relu(tf.matmul(tf.cast(self.query, tf.float32), weights) + bias)
                a = tf.einsum('ijk,kl->ijl', tf.cast(self.sim_text, tf.float32), weights)
                self.sim_text = tf.nn.relu(a + bias)
                b = tf.einsum('ijk,kl->ijl', tf.cast(self.unsim_text, tf.float32), weights)
                self.unsim_text = tf.nn.relu(b + bias)

                if is_normalize:
                    self.query = tf.layers.batch_normalization(self.query, training=is_trainning)
                    self.sim_text = tf.layers.batch_normalization(self.sim_text, training=is_trainning)
                    self.unsim_text = tf.layers.batch_normalization(self.unsim_text, training=is_trainning)

                L1_N = hidden_num

        # cos相似度计算
        with tf.name_scope("cos_similarity"):
            self.query = tf.expand_dims(self.query, axis=-1)
            self.similarity_1 = tf.einsum("ijk,ikl->ijl", self.sim_text, self.query)
            self.similarity_1 = tf.squeeze(self.similarity_1, axis=-1)
            # 对similarity_1进行mask操作
            self.similarity_1 = self.mask(self.similarity_1, maxsim_num, self.sim_text_num)

            self.similarity_2 = tf.einsum("ijk,ikl->ijl", self.unsim_text, self.query)
            self.similarity_2 = tf.squeeze(self.similarity_2, axis=-1)

        # 对similarity_1和similarity_2进行拼接 并softmax smoothing factor
        self.similarity = tf.concat([self.similarity_1, self.similarity_2], 1)
        self.smoothing_factor = tf.Variable(1.0, name='smoothing_factor')
        self.similarity = self.smoothing_factor * self.similarity
        self.logit = tf.nn.softmax(self.similarity)

        # 计算损失 只计算sim_text的损失
        self.loss = tf.log(self.logit)
        l = tf.cast(tf.sequence_mask(self.sim_text_num, maxsim_num + unsim_num), tf.float32)
        self.loss = tf.multiply(self.loss, l)
        self.cost = -tf.reduce_sum(self.loss, [0, 1])

        # 只在训练模型时定义反向传播操作
        if not is_trainning:
            return

        # 梯度更新
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def mask(self, Q, maxsim_num, length):
        masks = tf.cast(tf.sequence_mask(length, maxsim_num), tf.float32)
        Q = Q - (1 - masks) * tf.exp(10.0)
        return Q


if __name__ == '__main__':
    dssm = DSSM(10000, 3, 2, [300, 300, 128], True, True)
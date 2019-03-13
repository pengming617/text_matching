import tensorflow as tf


class PairCNN(object):
    def __init__(self, is_trainning, max_len, vocab_size, embedding_size, filter_sizes, num_filters, num_hidden,
                 learning_rate, k=2, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.text_a = tf.placeholder(tf.int32, [None, max_len], name="text_a")
        self.text_b = tf.placeholder(tf.int32, [None, max_len], name="text_b")
        self.y = tf.placeholder(tf.float32, [None, 2], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer for both CNN
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars_left = tf.expand_dims(tf.nn.embedding_lookup(W, self.text_a), -1)
            self.embedded_chars_right = tf.expand_dims(tf.nn.embedding_lookup(W, self.text_b), -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_left,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, max_len - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")

                # k_max_pooling over the outputs
                k_max_pooling = tf.nn.top_k(tf.transpose(h, [0, 3, 2, 1]), k=k, sorted=True)[0]
                k_max_pooling = tf.reshape(k_max_pooling, [-1, k*num_filters])
                # pooled_outputs_left.append(pooled)
                pooled_outputs_left.append(k_max_pooling)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_right,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, max_len - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")

                # k_max_pooling over the outputs
                k_max_pooling = tf.nn.top_k(tf.transpose(h, [0, 3, 2, 1]), k=k, sorted=True)[0]
                k_max_pooling = tf.reshape(k_max_pooling, [-1, k * num_filters])
                # pooled_outputs_right.append(pooled)
                pooled_outputs_right.append(k_max_pooling)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) * k
        self.h_pool_left = tf.reshape(tf.concat(pooled_outputs_left, 1), [-1, num_filters_total], name='h_pool_left')
        self.h_pool_right = tf.reshape(tf.concat(pooled_outputs_right, 1), [-1, num_filters_total], name='h_pool_right')

        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.h_pool_left, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.h_pool_right), 1, keep_dims=True)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Make input for classification
        self.new_input = tf.concat([self.h_pool_left, self.sims, self.h_pool_right], 1, name='new_input')

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[2*num_filters_total+1, num_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        if not is_trainning:
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    p = PairCNN(True, 20, 10000, 300, [3, 5], 100, 300, 0.001, 1, l2_reg_lambda=0.001)
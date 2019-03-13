import tensorflow as tf
from bimpm import layer_utils

eps = 1e-6


def cosine_distance(y1, y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm


def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1)  # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2)  # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,
                                       in_passage_repres_tmp)  # [batch_size, passage_len, question_len]
    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix


def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=2)  # [batch_size, passage_len, 'x', dim]
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0)  # [1, 1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)  # [batch_size, passage_len, decompse_dim, dim]


def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1)  # [batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0)  # [1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)  # [batch_size, decompse_dim, dim]


def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]

    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params)  # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params)  # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1)  # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0)  # [1, question_len, decompose_dim, dim]
        return cosine_distance(p, q)  # [passage_len, question_len, decompose]

    elems = (passage_rep, question_rep)
    matching_matrix = tf.map_fn(singel_instance, elems,
                                dtype=tf.float32)  # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix,
                                                                                            axis=2)])  # [batch_size, passage_len, 2*decompse_dim]


def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

    #     xdev = x - x.max()
    #     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev), -1)), -1))
    #     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask)  # [batch_size, passage_len]
    return tf.multiply(-1.0, tf.reduce_sum(result, -1))  # [batch_size]


def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    #     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans * gate + in_val * (1.0 - gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in range(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val


def cal_max_question_representation(question_representation, atten_scores):
    atten_positions = tf.argmax(atten_scores, axis=2, output_type=tf.int32)  # [batch_size, passage_len]
    max_question_reps = layer_utils.collect_representation(question_representation, atten_positions)
    return max_question_reps


def multi_perspective_match(feature_dim, repres1, repres2, is_training=True, dropout_rate=0.2,
                            options=None, scope_name='mp-match', reuse=False):
    '''
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    '''
    input_shape = tf.shape(repres1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        if options.with_cosine:
            cosine_value = layer_utils.cosine_distance(repres1, repres2, cosine_norm=False)
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            matching_result.append(cosine_value)
            match_dim += 1

        if options.with_mp_cosine:
            mp_cosine_params = tf.get_variable("mp_cosine", shape=[options.cosine_MP_dim, feature_dim],
                                               dtype=tf.float32)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            repres1_flat = tf.expand_dims(repres1, axis=2)
            repres2_flat = tf.expand_dims(repres2, axis=2)
            mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(repres1_flat, mp_cosine_params),
                                                             repres2_flat, cosine_norm=False)
            matching_result.append(mp_cosine_matching)
            match_dim += options.cosine_MP_dim

    matching_result = tf.concat(axis=2, values=matching_result)
    return (matching_result, match_dim)


def match_passage_with_question(passage_reps, question_reps, passage_mask, question_mask, passage_lengths,
                                question_lengths,
                                context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True,
                                with_max_attentive_match=True,
                                is_training=True, options=None, dropout_rate=0, forward=True):
    passage_reps = tf.multiply(passage_reps, tf.expand_dims(passage_mask, -1))
    question_reps = tf.multiply(question_reps, tf.expand_dims(question_mask, -1))
    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        relevancy_matrix = cal_relevancy_matrix(question_reps, passage_reps)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask)
        # relevancy_matrix = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
        #             scope_name="fw_attention", att_type=options.att_type, att_dim=options.att_dim,
        #             remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)

        all_question_aware_representatins.append(tf.reduce_max(relevancy_matrix, axis=2, keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(relevancy_matrix, axis=2, keep_dims=True))
        dim += 2
        if with_full_match:
            if forward:
                question_full_rep = layer_utils.collect_final_step_of_lstm(question_reps, question_lengths - 1)
            else:
                question_full_rep = question_reps[:, 0, :]

            passage_len = tf.shape(passage_reps)[1]
            question_full_rep = tf.expand_dims(question_full_rep, axis=1)
            question_full_rep = tf.tile(question_full_rep,
                                        [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]

            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                                                 passage_reps, question_full_rep,
                                                                 is_training=is_training,
                                                                 dropout_rate=options.dropout_rate,
                                                                 options=options, scope_name='mp-match-full-match')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_maxpool_match:
            maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                       shape=[options.cosine_MP_dim, context_lstm_dim],
                                                       dtype=tf.float32)
            maxpooling_rep = cal_maxpooling_matching(passage_reps, question_reps, maxpooling_decomp_params)
            all_question_aware_representatins.append(maxpooling_rep)
            dim += 2 * options.cosine_MP_dim

        if with_attentive_match:
            atten_scores = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim,
                                                          context_lstm_dim,
                                                          scope_name="attention", att_type=options.att_type,
                                                          att_dim=options.att_dim,
                                                          remove_diagnoal=False, mask1=passage_mask,
                                                          mask2=question_mask, is_training=is_training,
                                                          dropout_rate=dropout_rate)
            att_question_contexts = tf.matmul(atten_scores, question_reps)
            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                                                 passage_reps, att_question_contexts,
                                                                 is_training=is_training,
                                                                 dropout_rate=options.dropout_rate,
                                                                 options=options, scope_name='mp-match-att_question')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_max_attentive_match:
            max_att = cal_max_question_representation(question_reps, relevancy_matrix)
            (max_attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                                                     passage_reps, max_att, is_training=is_training,
                                                                     dropout_rate=options.dropout_rate,
                                                                     options=options, scope_name='mp-match-max-att')
            all_question_aware_representatins.append(max_attentive_rep)
            dim += match_dim

        all_question_aware_representatins = tf.concat(axis=2, values=all_question_aware_representatins)
    return (all_question_aware_representatins, dim)


def bilateral_match_func(in_question_repres, in_passage_repres,
                         question_lengths, passage_lengths, question_mask, passage_mask, input_dim, is_training,
                         options=None):
    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # ====word level matching======
    (match_reps, match_dim) = match_passage_with_question(in_passage_repres, in_question_repres, passage_mask,
                                                          question_mask, passage_lengths,
                                                          question_lengths, input_dim, scope="word_match_forward",
                                                          with_full_match=False,
                                                          with_maxpool_match=options.with_maxpool_match,
                                                          with_attentive_match=options.with_attentive_match,
                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                          is_training=is_training, options=options,
                                                          dropout_rate=options.dropout_rate, forward=True)
    question_aware_representatins.append(match_reps)
    question_aware_dim += match_dim

    (match_reps, match_dim) = match_passage_with_question(in_question_repres, in_passage_repres, question_mask,
                                                          passage_mask, question_lengths,
                                                          passage_lengths, input_dim, scope="word_match_backward",
                                                          with_full_match=False,
                                                          with_maxpool_match=options.with_maxpool_match,
                                                          with_attentive_match=options.with_attentive_match,
                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                          is_training=is_training, options=options,
                                                          dropout_rate=options.dropout_rate, forward=False)
    passage_aware_representatins.append(match_reps)
    passage_aware_dim += match_dim

    with tf.variable_scope('context_MP_matching'):
        for i in range(options.context_layer_num):  # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                # contextual lstm for both passage and question
                in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
                in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(passage_mask, axis=-1))
                (question_context_representation_fw, question_context_representation_bw,
                 in_question_repres) = layer_utils.my_lstm_layer(
                    in_question_repres, options.context_lstm_dim, input_lengths=question_lengths,
                    scope_name="context_represent",
                    reuse=False, is_training=is_training, dropout_rate=options.dropout_rate,
                    use_cudnn=options.use_cudnn)
                (passage_context_representation_fw, passage_context_representation_bw,
                 in_passage_repres) = layer_utils.my_lstm_layer(
                    in_passage_repres, options.context_lstm_dim, input_lengths=passage_lengths,
                    scope_name="context_represent",
                    reuse=True, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)

                # Multi-perspective matching
                with tf.variable_scope('left_MP_matching'):
                    (match_reps, match_dim) = match_passage_with_question(passage_context_representation_fw,
                                                                          question_context_representation_fw,
                                                                          passage_mask, question_mask, passage_lengths,
                                                                          question_lengths, options.context_lstm_dim,
                                                                          scope="forward_match",
                                                                          with_full_match=options.with_full_match,
                                                                          with_maxpool_match=options.with_maxpool_match,
                                                                          with_attentive_match=options.with_attentive_match,
                                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options.dropout_rate,
                                                                          forward=True)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim
                    (match_reps, match_dim) = match_passage_with_question(passage_context_representation_bw,
                                                                          question_context_representation_bw,
                                                                          passage_mask, question_mask, passage_lengths,
                                                                          question_lengths, options.context_lstm_dim,
                                                                          scope="backward_match",
                                                                          with_full_match=options.with_full_match,
                                                                          with_maxpool_match=options.with_maxpool_match,
                                                                          with_attentive_match=options.with_attentive_match,
                                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options.dropout_rate,
                                                                          forward=False)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim

                with tf.variable_scope('right_MP_matching'):
                    (match_reps, match_dim) = match_passage_with_question(question_context_representation_fw,
                                                                          passage_context_representation_fw,
                                                                          question_mask, passage_mask, question_lengths,
                                                                          passage_lengths, options.context_lstm_dim,
                                                                          scope="forward_match",
                                                                          with_full_match=options.with_full_match,
                                                                          with_maxpool_match=options.with_maxpool_match,
                                                                          with_attentive_match=options.with_attentive_match,
                                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options.dropout_rate,
                                                                          forward=True)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim
                    (match_reps, match_dim) = match_passage_with_question(question_context_representation_bw,
                                                                          passage_context_representation_bw,
                                                                          question_mask, passage_mask, question_lengths,
                                                                          passage_lengths, options.context_lstm_dim,
                                                                          scope="backward_match",
                                                                          with_full_match=options.with_full_match,
                                                                          with_maxpool_match=options.with_maxpool_match,
                                                                          with_attentive_match=options.with_attentive_match,
                                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                                          is_training=is_training, options=options,
                                                                          dropout_rate=options.dropout_rate,
                                                                          forward=False)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim

    question_aware_representatins = tf.concat(axis=2,
                                              values=question_aware_representatins)  # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(axis=2,
                                             values=passage_aware_representatins)  # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - options.dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - options.dropout_rate))

    # ======Highway layer======
    if options.with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,
                                                                options.highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,
                                                               options.highway_layer_num)

    # ========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0

    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in range(options.aggregation_layer_num):  # support multiple aggregation layer
            qa_aggregation_input = tf.multiply(qa_aggregation_input, tf.expand_dims(passage_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                qa_aggregation_input, options.aggregation_lstm_dim, input_lengths=passage_lengths,
                scope_name='left_layer-{}'.format(i),
                reuse=False, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, passage_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2 * options.aggregation_lstm_dim
            qa_aggregation_input = cur_aggregation_representation  # [batch_size, passage_len, 2*aggregation_lstm_dim]

            pa_aggregation_input = tf.multiply(pa_aggregation_input, tf.expand_dims(question_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                pa_aggregation_input, options.aggregation_lstm_dim,
                input_lengths=question_lengths, scope_name='right_layer-{}'.format(i),
                reuse=False, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, question_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2 * options.aggregation_lstm_dim
            pa_aggregation_input = cur_aggregation_representation  # [batch_size, passage_len, 2*aggregation_lstm_dim]

    aggregation_representation = tf.concat(axis=1, values=aggregation_representation)  # [batch_size, aggregation_dim]

    # ======Highway layer======
    if options.with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim,
                                                             options.highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])

    return (aggregation_representation, aggregation_dim)

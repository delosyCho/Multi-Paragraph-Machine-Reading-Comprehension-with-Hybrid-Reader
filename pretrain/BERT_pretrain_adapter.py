from modeling import create_attention_mask_from_input_mask

import modeling as modeling
import tensorflow as tf

import DataHolder_Pretrain as DataHolder
from utils import Fully_Connected
import numpy as np

import optimization

from attention_utils import compute_sparse_attention_mask
from modeling import get_shape_list


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    Examples
    ---------
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars


class KoNET:
    def __init__(self, firstTraining, testCase=False):
        self.first_training = firstTraining

        # self.save_path = '/home/ai/pycharm2/roberta_table/model.ckpt'
        self.save_path_cs = '/data/access_paper_models/adapter_pretrain_cs2/adapter_lm.ckpt'
        self.save_path_lm = '/data/access_paper_models/adapter_pretrain_lm2/adapter_lm.ckpt'
        self.save_path_rtp = '/data/access_paper_models/adapter_pretrain_rtp3/adapter_lm.ckpt'

        self.bert_path = '/data/MRC_models/Bigbird_KorQuAD1/mrc_model.ckpt'

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_segments = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_cols = tf.placeholder(shape=[None, None], dtype=tf.int32, name='cols')
        self.input_rows = tf.placeholder(shape=[None, None], dtype=tf.int32, name='rows')
        self.POS_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_weights = tf.placeholder(shape=[None], dtype=tf.float32)

        self.next_sentence_label = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.label_position = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.label_weight = tf.placeholder(shape=[None, None], dtype=tf.float32)
        self.label_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.label_rtp = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.label_se = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.label_rows = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.label_cols = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.start_label = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.stop_label = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.processor = DataHolder.DataHolder()
        self.keep_prob = 0.9
        if testCase is True:
            self.keep_prob = 1.0

        self.testCase = testCase

    def get_masked_lm_output(self, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights):
        """Get loss and log probs for the masked LM."""
        input_tensor = gather_indexes(input_tensor, positions)

        with tf.variable_scope("adapter_structure"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

        return loss, per_example_loss, log_probs

    def get_next_sentence_output(self, bert_config, input_tensor, labels):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[2, bert_config.hidden_size],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            # labels = tf.reshape(labels, [-1])
            # one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, per_example_loss, log_probs

    def create_model(self, input_ids, is_training=True, reuse=False, scope_name='bert'):
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')

        if self.testCase is True:
            is_training = False

        input_mask = tf.where(input_ids > 0, tf.ones_like(input_ids), tf.zeros_like(input_ids))
        attention_mask = compute_sparse_attention_mask(
            segment_ids=self.input_segments,
            column_ids=self.input_cols,
            row_ids=self.input_rows,
            input_mask=input_mask
        )

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            token_type_ids=self.input_segments,
            input_mask=input_mask,
            scope='bert',
            attention_mask=attention_mask
        )

        bert_variables = tf.global_variables()

        return model, bert_variables, model.get_sequence_output()

    def Table_Memory_Network(self, sequence_output, hidden_size=768, hops=1, dropout=0.2):
        # sequence_output = sequence_output + space_states
        row_one_hot = tf.one_hot(self.input_rows, depth=100)
        row_one_hot = tf.transpose(row_one_hot, perm=[0, 2, 1])

        column_one_hot = tf.one_hot(self.input_cols, depth=50)
        column_one_hot = tf.transpose(column_one_hot, perm=[0, 2, 1])

        column_wise_memory = tf.matmul(column_one_hot, sequence_output)
        row_wise_memory = tf.matmul(row_one_hot, sequence_output)

        reuse = False

        with tf.variable_scope("table_output_layer"):
            with tf.variable_scope("tab_mem"):
                for h in range(hops):
                    print('hop:', h)
                    with tf.variable_scope("column_memory_block", reuse=reuse):
                        column_wise_memory = modeling.attention_layer(
                            from_tensor=column_wise_memory,
                            to_tensor=sequence_output,
                            attention_mask=column_one_hot,
                        )

                    column_wise_memory = Fully_Connected(column_wise_memory, hidden_size, 'hidden_col' + str(0), gelu,
                                                         reuse=reuse)
                    column_wise_memory = modeling.dropout(column_wise_memory, dropout)

                    with tf.variable_scope("row_memory_block", reuse=reuse):
                        row_wise_memory = modeling.attention_layer(
                            from_tensor=row_wise_memory,
                            to_tensor=sequence_output,
                            attention_mask=row_one_hot)

                    row_wise_memory = Fully_Connected(row_wise_memory, hidden_size, 'hidden_row' + str(0), gelu,
                                                      reuse=reuse)
                    row_wise_memory = modeling.dropout(row_wise_memory, dropout)

                    reuse = True

        return column_wise_memory, row_wise_memory

    def transformer_layer(self,
                          layer_input,
                          hidden_size=768,
                          attention_heads=12,
                          n_layer=2,
                          name='adapter', dropout=True):
        input_ids = self.input_ids
        input_mask = tf.where(input_ids > 0, tf.ones_like(input_ids), tf.zeros_like(input_ids))
        attention_mask = compute_sparse_attention_mask(
            segment_ids=self.input_segments,
            column_ids=self.input_cols,
            row_ids=self.input_rows,
            input_mask=input_mask
        )

        with tf.variable_scope(name):
            if dropout is True:
                layer_output = modeling.transformer_model(layer_input,
                                                          num_hidden_layers=n_layer,
                                                          hidden_size=hidden_size,
                                                          num_attention_heads=attention_heads,
                                                          attention_mask=attention_mask)
            else:
                layer_output = modeling.transformer_model(layer_input,
                                                          num_hidden_layers=n_layer,
                                                          hidden_size=hidden_size,
                                                          num_attention_heads=attention_heads,
                                                          attention_probs_dropout_prob=0.0,
                                                          hidden_dropout_prob=0.0,
                                                          attention_mask=attention_mask)

        return layer_output

    def structured_adapter(self, embed_output, all_layer_outputs, is_training=True, return_layer_outputs=False):
        dropout = 0.0
        if is_training is True:
            dropout = 0.2

        former_output = embed_output
        layer_outputs = [all_layer_outputs[4], all_layer_outputs[8], all_layer_outputs[-1]]

        adapter_outputs = []

        with tf.variable_scope('structured_adapter'):
            for lx, layer_output in enumerate(layer_outputs):
                with tf.variable_scope('layer' + str(lx)):
                    column_memory, row_memory = self.Table_Memory_Network(layer_output, hops=2, dropout=dropout)
                    row_one_hot = tf.one_hot(self.input_rows, depth=100)
                    column_one_hot = tf.one_hot(self.input_cols, depth=50)

                    column_memory = tf.matmul(column_one_hot, column_memory)
                    row_memory = tf.matmul(row_one_hot, row_memory)

                    adapter_input = tf.concat([column_memory, row_memory, layer_output, former_output], axis=2)
                    adapter_input = Fully_Connected(adapter_input, 768, 'down_projection_layer_' + str(lx), gelu,
                                                    reuse=False)
                    adapter_output = self.transformer_layer(adapter_input, name='adapter_' + str(lx),
                                                            dropout=is_training)
                    former_output += adapter_output
                    former_output = gelu(former_output)
                    adapter_outputs.append(former_output)

        if return_layer_outputs is True:
            return former_output, adapter_outputs

        return former_output

    def joint_adapter(self, all_layer_outputs, structured_adapter_outputs, is_training=True):
        layer_outputs = [all_layer_outputs[8], all_layer_outputs[-1]]
        layer_outputs2 = [structured_adapter_outputs[1], structured_adapter_outputs[2]]

        former_output = structured_adapter_outputs[0]

        with tf.variable_scope('structured_adapter'):
            for lx, layer_output in enumerate(layer_outputs):
                with tf.variable_scope('layer' + str(lx)):
                    adapter_input = tf.concat([layer_outputs[lx], layer_outputs2[lx], former_output], axis=2)
                    adapter_input = Fully_Connected(adapter_input, 768, 'down_projection_layer_' + str(lx), gelu,
                                                    reuse=False)
                    adapter_output = self.transformer_layer(adapter_input, name='adapter_' + str(lx),
                                                            dropout=is_training)
                    former_output += adapter_output
                    former_output = gelu(former_output)

        return former_output

    def get_qa_loss(self, logit1, logit2, label1, label2):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit1, labels=label1)
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit2, labels=label2)

            loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)
        return loss, loss1, loss2

    def get_binary_probs(self, model_output, is_training=False, name='classification'):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.8

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope(name):
            with tf.variable_scope("entailment_block"):
                model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
                model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                model_output = Fully_Connected(model_output, output=768, name='hidden2', activation=gelu)
                model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                model_output = Fully_Connected(model_output, output=768, name='hidden3', activation=gelu)
                model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                model_output = Fully_Connected(model_output, output=512, name='hidden', activation=gelu)

            with tf.variable_scope("pointer_net"):
                log_probs = Fully_Connected(model_output, output=2, name='binary_prediction', activation=None, reuse=False)

        return log_probs

    def get_qa_probs(self, model_output, is_training=False, name='classification'):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.8

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope(name):
            with tf.variable_scope("adapter_structure"):
                with tf.variable_scope("MRC_block"):
                    model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
                    model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                    model_output = Fully_Connected(model_output, output=768, name='hidden2', activation=gelu)
                    model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                    model_output = Fully_Connected(model_output, output=768, name='hidden3', activation=gelu)
                    model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                    model_output = Fully_Connected(model_output, output=512, name='hidden', activation=gelu)

                with tf.variable_scope("pointer_net"):
                    log_probs = Fully_Connected(model_output, output=1, name='pointer_start1', activation=None, reuse=False)
                    log_probs = tf.squeeze(log_probs, axis=2)

        return log_probs

    def get_qa_probs2(self, model_output, is_training=False, name='classification'):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.8

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope(name):
            with tf.variable_scope("adapter_structure"):
                with tf.variable_scope("MRC_block"):
                    model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
                    model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                    model_output = Fully_Connected(model_output, output=768, name='hidden2', activation=gelu)
                    model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                    model_output = Fully_Connected(model_output, output=768, name='hidden3', activation=gelu)
                    model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

                    model_output = Fully_Connected(model_output, output=512, name='hidden', activation=gelu)

                with tf.variable_scope("pointer_net1"):
                    log_probs_s = Fully_Connected(model_output, output=1, name='pointer_start1', activation=None,
                                                  reuse=False)
                    log_probs_e = Fully_Connected(model_output, output=1, name='pointer_stop1', activation=None,
                                                  reuse=False)
                    log_probs_s = tf.squeeze(log_probs_s, axis=2)
                    log_probs_e = tf.squeeze(log_probs_e, axis=2)

                return log_probs_s, log_probs_e

    def get_table_embeddings(self, output, token_row_ids, token_col_ids, width=768):
        # Table Specific Embeddings
        input_shape = get_shape_list(token_row_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        with tf.variable_scope('adapter_embeddings'):
            bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')
            token_row_table = tf.get_variable(
                name='row_table',
                shape=[200, 768],
                initializer=create_initializer(bert_config.initializer_range * 0.3))

            token_col_table = tf.get_variable(
                name='col_table',
                shape=[200, 768],
                initializer=create_initializer(bert_config.initializer_range * 0.3))

        flat_token_row_ids = tf.reshape(token_row_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_row_ids, depth=200)
        token_row_embeddings = tf.matmul(one_hot_ids, token_row_table)
        token_row_embeddings = tf.reshape(token_row_embeddings,
                                          [batch_size, seq_length, width])
        output += token_row_embeddings

        flat_token_col_ids = tf.reshape(token_col_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_col_ids, depth=200)
        token_col_embeddings = tf.matmul(one_hot_ids, token_col_table)
        token_col_embeddings = tf.reshape(token_col_embeddings,
                                          [batch_size, seq_length, width])
        output += token_col_embeddings
        return output

    def training_lm(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')

        with tf.Session(config=config) as sess:
            model, bert_variables, seqeunce_output = self.create_model(self.input_ids, is_training=True)
            sequence_output = model.get_sequence_output()

            with tf.variable_scope('adapter_layers'):
                table_embeddings = self.get_table_embeddings(
                    output=model.get_embedding_output(),
                    token_row_ids=self.input_rows,
                    token_col_ids=self.input_cols
                )

                adapter_output = self.structured_adapter(table_embeddings, model.all_encoder_layers)
                column_memory, row_memory = self.Table_Memory_Network(adapter_output, hops=2, hidden_size=768)

                row_one_hot = tf.one_hot(self.input_rows, depth=100)
                column_one_hot = tf.one_hot(self.input_cols, depth=50)

                column_memory = tf.matmul(column_one_hot, column_memory)
                row_memory = tf.matmul(row_one_hot, row_memory)

                sequence_output = tf.concat([column_memory, row_memory, adapter_output, sequence_output], axis=2)

                input_tensor = sequence_output
                output_weights = model.get_embedding_table()
                positions = self.label_position
                label_ids = self.label_ids
                label_weights = self.label_weight

                loss1, _, _ = self.get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                                 label_ids, label_weights)

            loss = loss1

            learning_rate = 5e-6

            tvars = get_variables_with_name('adapter_layers')
            optimizer = optimization.create_optimizer(loss=loss, init_lr=learning_rate, num_train_steps=training_epoch,
                                                      num_warmup_steps=int(training_epoch * 0.1), use_tpu=False,
                                                      tvars=tvars)

            sess.run(tf.initialize_all_variables())

            bert_variables = get_variables_with_name('bert')

            saver = tf.train.Saver(bert_variables)
            saver.restore(sess, self.bert_path)
            print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path_lm)

            for i in range(training_epoch):
                input_ids_, input_mask, input_segments, input_row, input_col, label_ids_, label_position_, label_weight_ \
                = self.processor.next_batch()

                feed_dict = {self.input_ids: input_ids_, self.input_segments: input_segments,
                             self.input_rows: input_row, self.input_cols: input_col,
                             self.input_mask: input_mask,
                             self.label_position: label_position_, self.label_weight: label_weight_,
                             self.label_ids: label_ids_,}

                loss_, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
                print(i, np.array(loss_).shape)
                print(loss_)

                if i % 10000 == 0 and i != 0:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path_lm)

    def training_cs(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, is_training=True)

            with tf.variable_scope('adapter_layers'):
                table_embeddings = self.get_table_embeddings(
                    output=model.get_embedding_output(),
                    token_row_ids=self.input_rows,
                    token_col_ids=self.input_cols
                )

                adapter_output = self.structured_adapter(table_embeddings, model.all_encoder_layers)
                column_memory, row_memory = self.Table_Memory_Network(adapter_output, hops=2, hidden_size=768)

                row_one_hot = tf.one_hot(self.input_rows, depth=100)
                column_one_hot = tf.one_hot(self.input_cols, depth=50)

                column_memory = tf.matmul(column_one_hot, column_memory)
                row_memory = tf.matmul(row_one_hot, row_memory)

                sequence_output = tf.concat([column_memory, row_memory, adapter_output, sequence_output], axis=2)

            with tf.variable_scope('prediction_layers'):
                row_probs, col_probs = self.get_qa_probs2(sequence_output, is_training=True, name='row_prob')

            loss, loss_row, loss_col = self.get_qa_loss(row_probs, col_probs, self.label_rows, self.label_cols)

            learning_rate = 5e-6

            tvars = get_variables_with_name('adapter_layers')

            pvars = get_variables_with_name('prediction_layers')
            pvars.extend(tvars)

            optimizer = optimization.create_optimizer(loss=loss, init_lr=learning_rate, num_train_steps=135000,
                                                      num_warmup_steps=1000, use_tpu=False, tvars=pvars)

            sess.run(tf.initialize_all_variables())

            bert_variables.extend(tvars)
            saver = tf.train.Saver(bert_variables)
            saver.restore(sess, self.save_path_lm)
            print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path_cs)

            for i in range(training_epoch):
                input_ids_, input_segments, input_row, input_col, label_rows, label_cols \
                    = self.processor.next_batch_cs2()

                feed_dict = {self.input_ids: input_ids_, self.input_segments: input_segments,
                             self.input_rows: input_row, self.input_cols: input_col,
                             self.label_rows: label_rows, self.label_cols: label_cols
                              }

                loss_, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
                print(i, np.array(loss_).shape)
                print(loss_)

                if i % 10000 == 0 and i != 0:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path_cs)

    def training_rtp(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, is_training=True)

            with tf.variable_scope('adapter_layers'):
                table_embeddings = self.get_table_embeddings(
                    output=model.get_embedding_output(),
                    token_row_ids=self.input_rows,
                    token_col_ids=self.input_cols
                )

                adapter_output, adapter_outputs = self.structured_adapter(table_embeddings, model.all_encoder_layers,
                                                                          return_layer_outputs=True)
                column_memory, row_memory = self.Table_Memory_Network(adapter_output, hops=2, hidden_size=768)

                row_one_hot = tf.one_hot(self.input_rows, depth=100)
                column_one_hot = tf.one_hot(self.input_cols, depth=50)

                column_memory = tf.matmul(column_one_hot, column_memory)
                row_memory = tf.matmul(row_one_hot, row_memory)

                sequence_output = tf.concat([column_memory, row_memory, adapter_output, sequence_output], axis=2)

            with tf.variable_scope('joint_layers'):
                joint_adapter_output = self.joint_adapter(model.all_encoder_layers, adapter_outputs)

            sequence_output = tf.concat([adapter_output, joint_adapter_output, sequence_output], axis=2)
            pooled_output = tf.squeeze(sequence_output[:, 0:1, :], axis=1)

            with tf.variable_scope('joint_prediction_layers'):
                rtp_outputs = self.get_binary_probs(pooled_output, name='rtp_prediction_layer')
                se_outputs = self.get_binary_probs(pooled_output, name='se_prediction_layer')

            _, loss_rtp, loss_se = self.get_qa_loss(rtp_outputs, se_outputs, self.label_rtp, self.label_se)

            loss_rtp = tf.reduce_mean(loss_rtp * self.input_weights)
            loss_se = tf.reduce_mean(loss_se * (tf.ones_like(self.input_weights) - self.input_weights))
            total_loss = loss_rtp + loss_se
            learning_rate = 1e-5

            tvars = get_variables_with_name('adapter_layers')
            jvars = get_variables_with_name('joint_layers')
            pvars = get_variables_with_name('joint_prediction_layers')
            pvars.extend(jvars)

            optimizer = optimization.create_optimizer(loss=total_loss, init_lr=learning_rate, num_train_steps=135000,
                                                      num_warmup_steps=10000, use_tpu=False, tvars=pvars)

            sess.run(tf.initialize_all_variables())

            bert_variables.extend(tvars)
            saver = tf.train.Saver(bert_variables)
            saver.restore(sess, self.save_path_cs)
            print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path_cs)

            for i in range(training_epoch):
                input_ids, input_segments, input_cols, input_rows, input_weights, label_rtp, label_se \
                    = self.processor.next_batch_entailment()

                feed_dict = {self.input_ids: input_ids, self.input_segments: input_segments,
                             self.input_rows: input_rows, self.input_cols: input_cols,
                             self.input_weights: input_weights,
                             self.label_rtp: label_rtp, self.label_se: label_se
                              }

                loss_, loss_rtp_, loss_se_, _ = sess.run([total_loss, loss_rtp, loss_se, optimizer], feed_dict=feed_dict)
                print(i, np.array(loss_).shape)
                print(loss_, loss_rtp_, loss_se_)

                if i % 10000 == 0 and i != 0:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path_rtp)

    def training_rtp2(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, is_training=True)
            pooled_output = model.get_pooled_output()

            with tf.variable_scope('joint_prediction_layers'):
                rtp_outputs = self.get_binary_probs(pooled_output, name='rtp_prediction_layer')
                se_outputs = self.get_binary_probs(pooled_output, name='se_prediction_layer')

            _, loss_rtp, loss_se = self.get_qa_loss(rtp_outputs, se_outputs, self.label_rtp, self.label_se)

            loss_rtp = tf.reduce_mean(loss_rtp * self.input_weights)
            loss_se = tf.reduce_mean(loss_se * (tf.ones_like(self.input_weights) - self.input_weights))
            total_loss = loss_rtp + loss_se
            learning_rate = 1e-5

            tvars = get_variables_with_name('adapter_layers')
            jvars = get_variables_with_name('joint_layers')
            pvars = get_variables_with_name('joint_prediction_layers')
            pvars.extend(tvars)
            pvars.extend(jvars)

            optimizer = optimization.create_optimizer(loss=total_loss, init_lr=learning_rate, num_train_steps=135000,
                                                      num_warmup_steps=10000, use_tpu=False)

            sess.run(tf.initialize_all_variables())

            #bert_variables.extend(tvars)
            saver = tf.train.Saver(bert_variables)
            saver.restore(sess, self.bert_path)
            print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path_cs)

            for i in range(training_epoch):
                input_ids, input_segments, input_cols, input_rows, input_weights, label_rtp, label_se \
                    = self.processor.next_batch_entailment()

                feed_dict = {self.input_ids: input_ids, self.input_segments: input_segments,
                             self.input_rows: input_rows, self.input_cols: input_cols,
                             self.input_weights: input_weights,
                             self.label_rtp: label_rtp, self.label_se: label_se
                              }

                loss_, loss_rtp_, loss_se_, _ = sess.run([total_loss, loss_rtp, loss_se, optimizer], feed_dict=feed_dict)
                print(i, np.array(loss_).shape)
                print(loss_, loss_rtp_, loss_se_)

                if i % 100000 == 0 and i != 0:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path_rtp)


    def Training_tqa(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with tf.Session(config=config) as sess:
            input_mask = tf.where(self.input_ids > 0, tf.ones_like(self.input_ids), tf.zeros_like(self.input_ids))
            attention_mask = compute_sparse_attention_mask(self.input_segments, self.input_cols, self.input_rows,
                                                           input_mask)

            model, bert_variables, sequence_output = self.create_model(self.input_ids, is_training=True)
            adapter_output = self.structured_adapter(model.embedding_output, model.all_encoder_layers)

            column_memory, row_memory = self.Table_Memory_Network(adapter_output, hops=2, hidden_size=1024)

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)

            start_probs, stop_probs = self.get_qa_probs2(sequence_output, is_training=True, name='row_prob')

            loss, loss_row, loss_col = self.get_qa_loss(start_probs, stop_probs, self.start_label, self.stop_label)

            learning_rate = 1e-5

            tvars = get_variables_with_name('adapter_structure')
            optimizer = optimization.create_optimizer(loss=loss, init_lr=learning_rate, num_train_steps=135000,
                                                      num_warmup_steps=1000, use_tpu=False)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(bert_variables)
            saver.restore(sess, self.bert_path)
            print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path)

            for i in range(training_epoch):
                input_ids_, input_segments, input_row, input_col, start_label, stop_label \
                    = self.processor.next_batch_tqa()

                feed_dict = {self.input_ids: input_ids_, self.input_segments: input_segments,
                             self.input_rows: input_row, self.input_cols: input_col,
                             self.start_label: start_label, self.stop_label: stop_label
                              }

                loss_, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
                print(i, np.array(loss_).shape)
                print(loss_)

                if i % 10000 == 0 and i != 0:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path)

    def eval(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        bert_config = modeling.BertConfig.from_json_file('bert_config.json')

        with tf.Session(config=config) as sess:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.95

            bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base.json')

            with tf.Session(config=config) as sess:
                model, bert_variables, seqeunce_output = self.create_model(self.input_ids, self.input_mask,
                                                                           self.POS_ids)

                input_tensor = model.get_sequence_output()
                output_weights = model.get_embedding_table()
                positions = self.label_position
                label_ids = self.label_ids
                label_weights = self.label_weight
                next_label = self.next_sentence_label

                loss1, _, _ = self.get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                                                        label_ids, label_weights)

                input_tensor = model.get_pooled_output()
                loss2, _, _ = self.get_next_sentence_output(bert_config, input_tensor, next_label)

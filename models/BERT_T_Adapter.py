import utils.modeling as modeling
import tensorflow as tf

import utils.DataHolder2 as DataHolder2
from utils.utils import Fully_Connected
import utils.optimization as optimization
from utils.HTML_Utils import *
import utils.Chuncker as Chuncker
import utils.Table_Holder as Table_Holder
from utils.modeling import get_shape_list
from utils.attention_utils import compute_sparse_attention_mask

table_holder = Table_Holder.Holder()
chuncker = Chuncker.Chuncker()

def embedding_postprocessor(input_tensor,
                            col_ids,
                            row_type_ids,
                            hidden_size=768,
                            initializer_range=0.02, ):
    """Performs various post-processing on a word embedding tensor.

    Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
    float tensor with same shape as `input_tensor`.

    Raises:
    ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    # cols
    cols_table = tf.get_variable(
        name='col_embedding',
        shape=[50, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(col_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=50)
    token_type_embeddings = tf.matmul(one_hot_ids, cols_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

    # rows
    rows_table = tf.get_variable(
        name='row_embedding',
        shape=[250, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(row_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=250)
    token_type_embeddings = tf.matmul(one_hot_ids, rows_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

    return output

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

def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def masked_softmax(logits, mask, dim):
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


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
        self.chuncker = Chuncker.Chuncker()
        self.first_training = firstTraining
        #korquad
        #self.save_path = '/data/KorQuAD_Projects/tapas_kobigbird2/tapas_kobigbird.ckpt'

        self.save_path = '/data/access_paper_models/hybrid_adapter_model/qa_model.ckpt'
        #self.save_path = '/data/access_paper_models/hybrid_adapter_model2/qa_model.ckpt'
        self.bert_path = '/data/access_paper_models/adapter_pretrain_cs/adapter_lm.ckpt'

        #self.bert_path = '/home/ai/pycharm2/robert_adv_namu/bert_model.ckpt'
        #self.bert_path = '/data/pretrain_models/adapter_lm2/kotapas.ckpt'
        #self.bert_path = '/data/KorQuAD_Projects/dual_encoder_model2/dual_model.ckpt'

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_segments = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_cols = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input_cols')
        self.input_rows = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input_rows')
        self.input_names = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.start_label = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.stop_label = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.input_weights = tf.placeholder(shape=[None], dtype=tf.float32, name='input_weights')
        self.rank_weights = tf.placeholder(dtype=tf.float32, shape=[None], name='rank_weights')
        self.rank_label = tf.placeholder(dtype=tf.float32, shape=[None, None], name='rank_label')

        self.processor = DataHolder2.DataHolder()

        self.keep_prob = 0.9
        if testCase is True:
            self.keep_prob = 1.0

        self.testCase = testCase

        self.sess = None
        self.prediction_start = None
        self.prediction_stop = None

        self.column_size = 50
        self.row_size = 250

    def create_model(self, input_ids, input_segments, is_training=True, reuse=False, scope_name='bert'):
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
            input_mask=input_mask,
            token_type_ids=input_segments,
            scope=scope_name,
            reuse=reuse,
            attention_mask=attention_mask
        )

        bert_variables = tf.global_variables()

        return model, bert_variables, model.get_sequence_output()

    def get_verify_answer(self, model_output, is_training=False):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.85

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope("verification_block"):
            model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=512, name='hidden2', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=256, name='hidden3', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            log_probs = Fully_Connected(model_output, output=2, name='pointer_start1', activation=None, reuse=False)

        return log_probs

    def get_vf_loss(self, logit):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=self.rank_label)
        return loss

    def get_qa_loss2(self, logits, label):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label)
            loss = tf.reduce_mean(loss1)
        return loss

    def get_qa_loss_distill(self, start_probs, stop_probs):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=start_probs, labels=self.start_label)
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=stop_probs, labels=self.stop_label)

            start_label_dis = tf.nn.softmax(self.start_label_dis / 2)
            stop_label_dis = tf.nn.softmax(self.stop_label_dis / 2)

            loss3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=start_probs, labels=start_label_dis)
            loss4 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=stop_probs, labels=stop_label_dis)

            loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)
            loss_dis = tf.reduce_mean(loss3) + tf.reduce_mean(loss4)

        return loss * 0.7 + loss_dis * 0.3

    def get_qa_loss(self, logit1, logit2):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        with tf.variable_scope("cls/seq_relationship"):
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit1, labels=self.start_label)
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit1, labels=self.start_label)
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit2, labels=self.stop_label)

            loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)
        return loss, loss1, loss2

    def get_qa_probs(self, model_output, scope, is_training=False):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.8

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope("MRC_block_" + scope):
            model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=768, name='hidden2', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=768, name='hidden3', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=512, name='hidden', activation=gelu)

        with tf.variable_scope("pointer_net_" + scope):
            log_probs_s = Fully_Connected(model_output, output=1, name='pointer_start1', activation=None, reuse=False)
            log_probs_e = Fully_Connected(model_output, output=1, name='pointer_stop1', activation=None, reuse=False)
            log_probs_s = tf.squeeze(log_probs_s, axis=2)
            log_probs_e = tf.squeeze(log_probs_e, axis=2)

        return log_probs_s, log_probs_e

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

                    row_wise_memory = Fully_Connected(row_wise_memory, hidden_size, 'hidden_row' + str(0), gelu, reuse=reuse)
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

    def structured_adapter(self, embed_output, all_layer_outputs, is_training=True):
        dropout = 0.0
        if is_training is True:
            dropout = 0.2

        former_output = embed_output
        layer_outputs = [all_layer_outputs[0], all_layer_outputs[4], all_layer_outputs[8], all_layer_outputs[-1]]

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

        return former_output

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

    def Training(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=True,
                                                                       scope_name='bert')

            with tf.variable_scope('adapter_layers'):
                table_embeddings = self.get_table_embeddings(
                    output=model.get_embedding_output(),
                    token_row_ids=self.input_rows,
                    token_col_ids=self.input_cols
                )

                adapter_output = self.structured_adapter(table_embeddings, model.all_encoder_layers)

            with tf.variable_scope('output_layer_text'):
                column_memory, row_memory = self.Table_Memory_Network(sequence_output, hops=2, hidden_size=768)

                row_one_hot = tf.one_hot(self.input_rows, depth=100)
                column_one_hot = tf.one_hot(self.input_cols, depth=50)

                column_memory = tf.matmul(column_one_hot, column_memory)
                row_memory = tf.matmul(row_one_hot, row_memory)

                sequence_output = tf.concat([column_memory, row_memory, adapter_output, sequence_output], axis=2)
                probs_start2, probs_stop2 = self.get_qa_probs(sequence_output, scope='table_layer', is_training=True)

            loss2, _, _ = self.get_qa_loss(probs_start2, probs_stop2)
            qa_loss = loss2
            qa_loss = tf.reduce_mean(qa_loss)
            total_loss = qa_loss

            learning_rate = 2e-5

            tvars = get_variables_with_name('adapter_layers')
            tvars.extend(bert_variables)

            ovars = get_variables_with_name('output_layer')
            bert_variables.extend(ovars)

            optimizer = optimization.create_optimizer(loss=total_loss,
                                                      init_lr=learning_rate,
                                                      num_train_steps=training_epoch,
                                                      num_warmup_steps=int(training_epoch * 0.1),
                                                      use_tpu=False,
                                                      tvars=bert_variables)
            sess.run(tf.initialize_all_variables())

            if self.first_training is True:
                saver = tf.train.Saver(tvars)
                saver.restore(sess, self.bert_path)
                print('BERT restored')

            if is_Continue is True:
                saver = tf.train.Saver()
                saver.restore(sess, self.save_path)

            self.processor.batch_size = 3
            for i in range(training_epoch):
                input_ids, input_mask, input_segments, input_rows, input_cols, start_label, stop_label \
                    = self.processor.next_batch_tqa()

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: input_segments,
                             self.start_label: start_label, self.stop_label: stop_label,
                             self.input_rows: input_rows, self.input_cols: input_cols}

                loss_, _ = sess.run([qa_loss, optimizer], feed_dict=feed_dict)
                print(i, loss_)

                if i % 1000 == 0 and i > 100:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path)

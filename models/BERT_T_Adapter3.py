import utils.modeling as modeling
import tensorflow as tf

import utils.DataHolder2 as DataHolder2
import utils.DataHolder_test as DataHolder_test
from utils.utils import Fully_Connected
import utils.optimization as optimization
from utils.HTML_Utils import *
import utils.Chuncker as Chuncker
from utils.modeling import get_shape_list
from utils.attention_utils import compute_sparse_attention_mask

import numpy as np
import utils.tokenization as tokenization
from transformers import AutoTokenizer
import os
import json
from utils.evaluate2 import f1_score, exact_match_score
import utils.Ranking_ids as Ranking_ids

from utils.HTML_Processor import process_document
import utils.Table_Holder as Table_Holder

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
        self.save_path = '/data/KorQuAD_Projects/tapas_kobigbird2/tapas_kobigbird.ckpt'
        #self.bert_path = '/data/MRC_models/Bigbird_KorQuAD1/mrc_model.ckpt'

        self.bert_path = '/data/pretrain_models/adapter_lm1/kotapas.ckpt'

        # self.bert_path = '/home/ai/pycharm2/robert_adv_namu/bert_model.ckpt'
        #self.bert_path = '/data/pretrain_models/adapter_lm2/kotapas.ckpt'
        #self.bert_path = '/data/KorQuAD_Projects/dual_encoder_model2/dual_model.ckpt'

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_segments = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_positions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_cols = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_rows = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_names = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_rankings = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.query_mask = tf.placeholder(shape=[None, None], dtype=tf.float32)

        self.column_label = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.row_label = tf.placeholder(shape=[None, None], dtype=tf.int32)

        self.start_label = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.stop_label = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.start_label_dis = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.stop_label_dis = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.start_label2 = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.stop_label2 = tf.placeholder(dtype=tf.float32, shape=[None, None])
        self.rank_label = tf.placeholder(dtype=tf.float32, shape=[None, None])
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

    def create_model(self, input_ids, input_mask, input_segments, is_training=True, reuse=False, scope_name='bert'):
        bert_config = modeling.BertConfig.from_json_file('bert_config_mecab_base_rr.json')

        if self.testCase is True:
            is_training = False

        input_mask = tf.where(input_ids > 0, tf.ones_like(input_ids), tf.zeros_like(input_ids))
        attention_mask = compute_sparse_attention_mask(
            segment_ids=self.input_segments,
            column_ids=self.input_cols,
            row_ids=self.input_rows,
            input_mask=self.input_mask
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

    def get_qa_probs(self, model_output, is_training=False):
        """Get loss and log probs for the next sentence prediction."""

        keep_prob = 0.8

        if is_training is False:
            keep_prob = 1.0

        with tf.variable_scope("MRC_block"):
            model_output = Fully_Connected(model_output, output=768, name='hidden1', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=768, name='hidden2', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=768, name='hidden3', activation=gelu)
            model_output = tf.nn.dropout(model_output, keep_prob=keep_prob)

            model_output = Fully_Connected(model_output, output=512, name='hidden', activation=gelu)

        with tf.variable_scope("pointer_net1"):
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

    def adapter_layer(self, layer_input, name='adapter', dropout=True):
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
                layer_output = modeling.transformer_model(layer_input, num_hidden_layers=1, hidden_size=1024,
                                                          num_attention_heads=8, attention_mask=attention_mask)
            else:
                layer_output = modeling.transformer_model(layer_input, num_hidden_layers=1, hidden_size=1024,
                                                          num_attention_heads=8, attention_probs_dropout_prob=0.0,
                                                          hidden_dropout_prob=0.0, attention_mask=attention_mask)

        return layer_output

    def structured_adapter(self, embedding_output, all_layer_outputs, is_training=True):
        dropout = 0.0
        if is_training is True:
            dropout = 0.2

        layer_outputs = [all_layer_outputs[4], all_layer_outputs[8], all_layer_outputs[-1]]

        with tf.variable_scope('adapter_structure'):
            column_memory, row_memory = self.Table_Memory_Network(all_layer_outputs[0], hops=2, dropout=dropout)
            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)
            adapter_input = tf.concat([column_memory, row_memory, all_layer_outputs[0]], axis=2)
            adapter_input = Fully_Connected(adapter_input, 1024, 'adapter_layer_concat', gelu, reuse=False)

            adapter_output = self.adapter_layer(adapter_input, 'adapter_layer_input', is_training)

            for lx, layer_output in enumerate(layer_outputs):
                with tf.variable_scope('layer' + str(lx)):
                    column_memory, row_memory = self.Table_Memory_Network(layer_output, hops=2, dropout=dropout)
                    row_one_hot = tf.one_hot(self.input_rows, depth=100)
                    column_one_hot = tf.one_hot(self.input_cols, depth=50)

                    column_memory = tf.matmul(column_one_hot, column_memory)
                    row_memory = tf.matmul(row_one_hot, row_memory)
                    adapter_input = tf.concat([column_memory, row_memory, layer_output, adapter_output], axis=2)

                    adapter_input = Fully_Connected(adapter_input, 1024, 'adapter_layer_' + str(lx), gelu, reuse=False)
                    adapter_output = self.adapter_layer(adapter_input, name='adapter_' + str(lx), dropout=is_training)

        return adapter_output

    def Training(self, is_Continue, training_epoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids,
                                                                       self.input_segments, is_training=True,
                                                                       scope_name='bert')

            with tf.variable_scope('output_layer_vf'):
                pooled_output = model.get_pooled_output()
                probs_vf = self.get_verify_answer(pooled_output, is_training=True)

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
                probs_start2, probs_stop2 = self.get_qa_probs(sequence_output, is_training=True)

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
                ##
                # 라벨 정보 바뀌어야함
                ##
                input_ids, input_segments, start_label, stop_label, \
                input_rows, input_cols, vf_label, rank_weight \
                    = self.processor.next_batch_table_only()

                feed_dict = {self.input_ids: input_ids,
                             self.input_segments: input_segments,
                             self.start_label: start_label, self.stop_label: stop_label,
                             self.input_rows: input_rows, self.input_cols: input_cols,
                             }

                loss_, loss2_, _ = sess.run([qa_loss, vf_loss, optimizer], feed_dict=feed_dict)
                print(i, loss_, loss2_)

                if i % 1000 == 0 and i > 100:
                    print('saved!')
                    saver = tf.train.Saver()
                    saver.save(sess, self.save_path)

    def eval_with_span(self):
        def clean_tokenize(query):
            bert_tokens = tokenizer.tokenize(query)
            # print(bert_tokens)
            tokens = []
            pre_text = ""
            for i in range(len(bert_tokens)):
                bert_token = bert_tokens[i].replace("##", "")
                if i + 1 < len(bert_tokens):
                    post_token = bert_tokens[i + 1].replace("##", "")
                else:
                    post_token = ""
                if bert_token == '[UNK]':
                    token = str(
                        re.match(f"{pre_text}(.*){post_token}(.*)",
                                 query).group(1))
                    tokens.append(token)
                    pre_text += token
                else:
                    tokens.append(bert_token)
                    pre_text += bert_token
            return tokens

        file = open('qtype.csv', 'r', encoding='utf-8')
        lines = file.read().split('\n')

        queries = []
        codes = []

        for line in lines:
            tk = line.split(',')
            try:
                codes.append(int(tk[1]))
                queries.append(tk[0])
            except:
                continue

        # print(len(queries), len(codes))
        # input()

        # name_tagger = Name_Tagging.Name_tagger()
        chuncker = Chuncker.Chuncker()

        path_dir = '/data/korquad_data/korquad2_dev'

        file_list = os.listdir(path_dir)
        file_list.sort()

        vocab = tokenization.load_vocab(vocab_file='vocab_bigbird.txt')
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab_bigbird.txt')

        em_total = 0
        f1_total = 0
        epo = 0

        em_total1 = 0
        f1_total1 = 0
        epo1 = 0

        em_total2 = 0
        f1_total2 = 0
        epo2 = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, self.input_mask,
                                                                       self.input_segments, is_training=False,
                                                                       scope_name='bert')

            adapter_output = self.structured_adapter(model.embedding_output, model.all_encoder_layers,
                                                     is_training=False)
            with tf.variable_scope('output_layer'):
                column_memory, row_memory = self.Table_Memory_Network(adapter_output, hops=2, dropout=0.0, hidden_size=1024)
            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)
            prob_start, prob_stop = self.get_qa_probs(sequence_output, is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            num_tag = ['가장', '더', '제일', '많이', '적게']

            for file_name in file_list:
                print(file_name, 'processing evaluation')

                in_path = path_dir + '/' + file_name
                data = json.load(open(in_path, 'r', encoding='utf-8'))

                for article in data['data']:
                    doc = str(article['context'])

                    for qas in article['qas']:
                        error_code = -1

                        answer = qas['answer']
                        answer_start = answer['answer_start']
                        answer_text = answer['text']
                        question = qas['question']

                        chuncker.get_feautre(query=question)

                        if len(answer_text) > 40:
                            continue

                        query_tokens = []
                        query_tokens.append('[CLS]')
                        q_tokens = tokenizer.tokenize(question.lower())
                        for tk in q_tokens:
                            query_tokens.append(tk)
                        query_tokens.append('[SEP]')

                        ######
                        # 정답에 ans 토큰을 임베딩하기 위한 코드
                        ######

                        ans1 = ''
                        ans2 = ''
                        if doc[answer_start - 1] == ' ':
                            ans1 = '[answer]'
                        else:
                            ans1 = '[answer]'

                        if doc[answer_start + len(answer_text)] == ' ':
                            ans2 = '[/answer]'
                        else:
                            ans2 = '[/answer]'

                        doc_ = doc[0: answer_start] + ans1 + answer_text + ans2 + doc[
                                                                                  answer_start + len(answer_text): -1]
                        doc_ = str(doc_)
                        #
                        #####

                        paragraphs = doc_.split('<h2>')
                        sequences = []

                        checked = False

                        for paragraph in paragraphs:
                            try:
                                title = paragraph.split('[/h2]')[0]
                                paragraph = paragraph.split('[/h2]')[1]
                            except:
                                title = ''

                            sub_paragraphs = paragraph.split('<h3>')

                            for sub_paragraph in sub_paragraphs:
                                if checked is True:
                                    break

                                paragraph_, table_list = pre_process_document(paragraph, answer_setting=False,
                                                                              a_token1='',
                                                                              a_token2='')
                                paragraph = process_document(paragraph)

                                for table_text in table_list:
                                    if checked is True:
                                        break

                                    if table_text.find('[answer]') != -1:
                                        table_text = table_text.replace('[answer]', '')
                                        table_text = table_text.replace('[/answer]', '')

                                        table_text = table_text.replace('<th', '<td')
                                        table_text = table_text.replace('</th', '</td')

                                        table_text = table_text.replace(' <td>', '<td>')
                                        table_text = table_text.replace(' <td>', '<td>')
                                        table_text = table_text.replace('\n<td>', '<td>')
                                        table_text = table_text.replace('</td> ', '</td>')
                                        table_text = table_text.replace('</td> ', '</td>')
                                        table_text = table_text.replace('\n<td>', '<td>')
                                        table_text = table_text.replace('[answer]<td>', '<td>[answer] ')
                                        table_text = table_text.replace('</td>[/answer]', ' [/answer]</td>')
                                        table_text = table_text.replace('</td>', '  </td>')
                                        table_text = table_text.replace('<td>', '<td> ')

                                        table_text = table_text.replace('<a>', '')
                                        table_text = table_text.replace('<b>', '')
                                        table_text = table_text.replace('</a>', '')
                                        table_text = table_text.replace('</b>', '')

                                        table_text, child_texts = overlap_table_process(table_text=table_text)
                                        table_text = head_process(table_text=table_text)

                                        table_holder.get_table_text(table_text=table_text)
                                        table_data = table_holder.table_data
                                        lengths = []

                                        for data in table_data:
                                            lengths.append(len(data))
                                        if len(lengths) <= 0:
                                            break

                                        length = max(lengths)

                                        rank_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                                        col_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)
                                        row_ids = np.zeros(shape=[len(table_data), length], dtype=np.int32)

                                        count_arr = np.zeros(shape=[200], dtype=np.int32)
                                        for data in table_data:
                                            count_arr[len(data)] += 1
                                        table_head = get_table_head(table_text=table_text, count_arr=count_arr)

                                        rankings = Ranking_ids.numberToRanking(table_data, table_head)

                                        for j in range(length):
                                            for i in range(len(table_data)):
                                                col_ids[i, j] = j
                                                row_ids[i, j] = i
                                                rank_ids[i, j] = rankings[i][j]

                                        idx = 0
                                        tokens_ = []
                                        clean_tokens_ = []
                                        rows_ = []
                                        cols_ = []
                                        ranks_ = []
                                        name_tags_ = []

                                        for i in range(len(table_data)):
                                            for j in range(len(table_data[i])):
                                                if table_data[i][j] is not None:
                                                    tokens = tokenizer.tokenize(table_data[i][j])
                                                    try:
                                                        clean_tokens = clean_tokenize(str(table_data[i][j]).strip())
                                                    except:
                                                        print('error:', table_data[i][j])
                                                        clean_tokens = tokens
                                                    # name_tag = name_tagger.get_name_tag(table_data[i][j])

                                                    for k, tk in enumerate(tokens):
                                                        tokens_.append(tk)
                                                        clean_tokens_.append(clean_tokens[k])
                                                        rows_.append(i + 1)
                                                        cols_.append(j)
                                                        ranks_.append(rank_ids[i][j])
                                                        # name_tags_.append(name_tag)

                                                        if k >= 50:
                                                            break

                                                    if len(tokens) > 50 and str(table_data[i][j]).find(
                                                            '[/answer]') != -1:
                                                        tokens_.append('[/answer]')
                                                        rows_.append(i)
                                                        cols_.append(j)
                                                        ranks_.append(rank_ids[i][j])
                                                        # name_tags_.append(name_tag)
                                        # print(clean_tokens_)
                                        start_idx = -1
                                        end_idx = -1

                                        tokens = []
                                        clean_tokens = []
                                        rows = []
                                        cols = []
                                        ranks = []
                                        segments = []
                                        name_tags = []

                                        for tk in query_tokens:
                                            tokens.append(tk)
                                            clean_tokens.append(tk)
                                            rows.append(0)
                                            cols.append(0)
                                            ranks.append(0)
                                            segments.append(0)
                                            name_tags.append(0)

                                        for j, tk in enumerate(tokens_):
                                            if tk == '[answer]':
                                                start_idx = len(tokens)
                                            elif tk == '[/answer]':
                                                end_idx = len(tokens) - 1
                                            else:
                                                tokens.append(tk)
                                                clean_tokens.append(clean_tokens_[j])
                                                rows.append(rows_[j] + 1)
                                                cols.append(cols_[j] + 1)
                                                ranks.append(ranks_[j])
                                                segments.append(1)
                                                # name_tags.append(name_tags_[j])

                                        ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=tokens)

                                        # print(tokens)
                                        # input()

                                        max_length = 512

                                        length = len(ids)
                                        if length > max_length:
                                            length = max_length

                                        input_ids = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        input_mask = np.zeros(shape=[1, max_length], dtype=np.int32)

                                        segments_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        ranks_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        cols_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        rows_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)
                                        names_has_ans = np.zeros(shape=[1, max_length], dtype=np.int32)

                                        count = 0

                                        for j in range(length):
                                            input_ids[count, j] = ids[j]
                                            segments_has_ans[count, j] = segments[j]
                                            cols_has_ans[count, j] = cols[j]
                                            rows_has_ans[count, j] = rows[j]
                                            ranks_has_ans[count, j] = ranks[j]
                                            input_mask[count, j] = 1
                                            # names_has_ans[count, j] = name_tags[j]

                                        feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                                                     self.input_segments: segments_has_ans,
                                                     self.input_names: names_has_ans,
                                                     self.input_rankings: ranks_has_ans,
                                                     self.input_rows: rows_has_ans, self.input_cols: cols_has_ans}

                                        if input_ids.shape[0] > 0:

                                            probs_start, probs_stop = \
                                                sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                                            probs_start = np.array(probs_start, dtype=np.float32)
                                            probs_stop = np.array(probs_stop, dtype=np.float32)

                                            for j in range(input_ids.shape[0]):
                                                for k in range(1, input_ids.shape[1]):
                                                    probs_start[j, k] = 0
                                                    probs_stop[j, k] = 0

                                                    if input_ids[j, k] == 3:
                                                        break

                                            self.chuncker.get_feautre(question)

                                            prob_scores = []
                                            c_scores = []

                                            for j in range(input_ids.shape[0]):
                                                # paragraph ranking을 위한 score 산정기준
                                                # score2 = ev_values[j, 0]
                                                score2 = 2 - (probs_start[j, 0] + probs_stop[j, 0])

                                                prob_scores.append(score2)
                                                # c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                                            if True:
                                                for j in range(input_ids.shape[0]):
                                                    probs_start[j, 0] = -999
                                                    probs_stop[j, 0] = -999

                                                # CLS 선택 무효화

                                                prediction_start = probs_start.argmax(axis=1)
                                                prediction_stop = probs_stop.argmax(axis=1)

                                                answers = []
                                                scores = []
                                                candi_scores = []

                                                for j in range(input_ids.shape[0]):
                                                    answer_start_idx = prediction_start[j]
                                                    answer_stop_idx = prediction_stop[j]

                                                    if cols_has_ans[0, answer_start_idx] != cols_has_ans[
                                                        0, answer_stop_idx]:
                                                        answer_stop_idx2 = answer_stop_idx
                                                        answer_stop_idx = answer_start_idx
                                                        answer_start_idx2 = answer_stop_idx2

                                                        for k in range(answer_start_idx + 1, input_ids.shape[1]):
                                                            if cols_has_ans[0, k] == cols_has_ans[0, answer_start_idx]:
                                                                answer_stop_idx = k
                                                            else:
                                                                break

                                                        for k in reversed(list(range(0, answer_stop_idx2 - 1))):
                                                            if cols_has_ans[0, k] == cols_has_ans[0, answer_stop_idx2]:
                                                                answer_start_idx2 = k
                                                            else:
                                                                break

                                                        prob_1 = probs_start[0, answer_start_idx] + \
                                                                 probs_stop[0, answer_stop_idx]

                                                        prob_2 = probs_start[0, answer_start_idx2] + \
                                                                 probs_stop[0, answer_stop_idx2]

                                                        if prob_2 > prob_1:
                                                            answer_start_idx = answer_start_idx2
                                                            answer_stop_idx = answer_stop_idx2

                                                    score = probs_start[j, answer_start_idx]
                                                    scores.append(score * 1)
                                                    candi_scores.append(score * 1)

                                                    if answer_start_idx > answer_stop_idx:
                                                        answer_stop_idx = answer_start_idx + 15
                                                    if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                                        for k in range(answer_start_idx, input_ids.shape[1]):
                                                            if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                                                answer_stop_idx = k
                                                                break

                                                    answer = ''

                                                    if answer_stop_idx + 1 >= input_ids.shape[1]:
                                                        answer_stop_idx = input_ids.shape[1] - 2

                                                    probs1 = probs_start[0, answer_start_idx]
                                                    probs2 = probs_stop[0, answer_stop_idx]
                                                    new_start_idx = answer_start_idx
                                                    new_stop_idx = answer_stop_idx

                                                    for k in range(answer_start_idx, answer_stop_idx + 1):
                                                        if k < 512 - 1:
                                                            if cols_has_ans[0, k] != cols_has_ans[0, k + 1]:
                                                                probs1 += probs_stop[0, k]
                                                                new_stop_idx = k
                                                                break

                                                    for k in list(
                                                            reversed(range(answer_start_idx, answer_stop_idx + 1))):
                                                        if k > 1:
                                                            if cols_has_ans[0, k] != cols_has_ans[0, k - 1]:
                                                                probs2 += probs_start[0, k]
                                                                new_start_idx = k
                                                                break

                                                    if probs1 > probs2:
                                                        answer_stop_idx = new_stop_idx
                                                    else:
                                                        answer_start_idx = new_start_idx

                                                    for a_i, k in enumerate(
                                                            range(answer_start_idx, answer_stop_idx + 1)):
                                                        if a_i > 1:
                                                            if cols_has_ans[0, k] != cols_has_ans[0, k - 1]:
                                                                break
                                                        answer += str(clean_tokens[k])

                                                    answers.append(answer.replace(' ##', ''))

                                            if len(answers) > 0:
                                                answer_candidates = []
                                                candidates_scores = []

                                                for _ in range(1):
                                                    m_s = -99
                                                    m_ix = 0

                                                    for q in range(len(scores)):
                                                        if m_s < scores[q]:
                                                            m_s = scores[q]
                                                            m_ix = q

                                                    answer_candidates.append(answer_re_touch(answers[m_ix]))
                                                    candidates_scores.append(candi_scores[m_ix])
                                                    # print('score:', scores[m_ix])
                                                    # scores[m_ix] = -999

                                                a1 = [0]
                                                a2 = [0]

                                                for a_c in answer_candidates:
                                                    if a_c.find('<table>') != -1:
                                                        continue

                                                    a1.append(
                                                        exact_match_score(prediction=a_c, ground_truth=answer_text))
                                                    a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                                                for q in range(len(queries)):
                                                    if queries[q].strip() == question.strip():
                                                        code = codes[q]

                                                        if code == 1 or code == 2:
                                                            em_total1 += max(a1)
                                                            f1_total1 += max(a2)
                                                            epo1 += 1
                                                        else:
                                                            em_total2 += max(a1)
                                                            f1_total2 += max(a2)
                                                            epo2 += 1

                                                if epo1 > 0 and epo2 > 0:
                                                    print('EM1:', em_total1 / epo1)
                                                    print('F11:', f1_total1 / epo1)
                                                    print()
                                                    print('EM2:', em_total2 / epo2)
                                                    print('F12:', f1_total2 / epo2)

                                                em_total += max(a1)
                                                f1_total += max(a2)
                                                epo += 1

                                                for j in range(input_ids.shape[0]):
                                                    check = 'None'
                                                    answer_text = answer_text.replace('<a>', '').replace('</a>', '')

                                                    f1_ = f1_score(prediction=answer_re_touch(answers[j]),
                                                                   ground_truth=answer_text)

                                                    text = answers[j]

                                                    print('score:', scores[j], check, type, 'F1:',
                                                          f1_, ' , ',
                                                          text.replace('\n', ' '))
                                                print(table_text.replace('\n', ''))
                                                print('question:', question)
                                                print('answer:', answer_text)
                                                print('EM:', em_total / epo)
                                                print('F1:', f1_total / epo)
                                                print('-----\n', epo)

    def eval_with_span_2(self):
        self.keep_prob = 1.0

        file = open('qtype.csv', 'r', encoding='utf-8')
        lines = file.read().split('\n')

        queries = []
        codes = []

        for line in lines:
            tk = line.split(',')
            try:
                codes.append(int(tk[1]))
                queries.append(tk[0])
            except:
                continue

        # print(len(queries), len(codes))
        # input()

        # name_tagger = Name_Tagging.Name_tagger()
        chuncker = Chuncker.Chuncker()

        path_dir = '/data/korquad_data/korquad2_dev'

        file_list = os.listdir(path_dir)
        file_list.sort()

        vocab = tokenization.load_vocab(vocab_file='vocab.txt')
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt')

        em_total = 0
        f1_total = 0
        epo = 0

        em_total1 = 0
        f1_total1 = 0
        epo1 = 0

        em_total2 = 0
        f1_total2 = 0
        epo2 = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, self.input_mask,
                                                                       self.input_segments, is_training=False)

            input_shape = get_shape_list(sequence_output, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]

            column_memory, row_memory = self.Table_Memory_Network(sequence_output=sequence_output, hops=3, dropout=0.0)

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)

            prob_start, prob_stop = self.get_qa_probs(sequence_output, is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)
            max_length = 512

            for _ in range(1618):
                input_ids, input_mask, segments_has_ans, rows_has_ans, cols_has_ans, answer_text = \
                    self.processor.next_batch_test()

                feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                             self.input_segments: segments_has_ans,
                             # self.input_names: names_has_ans,
                             # self.input_rankings: ranks_has_ans,
                             self.input_rows: rows_has_ans, self.input_cols: cols_has_ans}

                if input_ids.shape[0] > 0:

                    probs_start, probs_stop = \
                        sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                    probs_start = np.array(probs_start, dtype=np.float32)
                    probs_stop = np.array(probs_stop, dtype=np.float32)

                    for j in range(input_ids.shape[0]):
                        for k in range(1, input_ids.shape[1]):
                            probs_start[j, k] = 0
                            probs_stop[j, k] = 0

                            if input_ids[j, k] == 3:
                                break

                    prob_scores = []
                    c_scores = []

                    for j in range(input_ids.shape[0]):
                        # paragraph ranking을 위한 score 산정기준
                        # score2 = ev_values[j, 0]
                        score2 = 2 - (probs_start[j, 0] + probs_stop[j, 0])

                        prob_scores.append(score2)
                        # c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                    if True:
                        for j in range(input_ids.shape[0]):
                            probs_start[j, 0] = -999
                            probs_stop[j, 0] = -999

                        # CLS 선택 무효화

                        prediction_start = probs_start.argmax(axis=1)
                        prediction_stop = probs_stop.argmax(axis=1)

                        answers = []
                        scores = []
                        candi_scores = []

                        for j in range(input_ids.shape[0]):
                            answer_start_idx = prediction_start[j]
                            answer_stop_idx = prediction_stop[j]

                            if cols_has_ans[0, answer_start_idx] != cols_has_ans[0, answer_stop_idx]:
                                answer_stop_idx2 = answer_stop_idx
                                answer_stop_idx = answer_start_idx
                                answer_start_idx2 = answer_stop_idx2

                                for k in range(answer_start_idx + 1, input_ids.shape[1]):
                                    if cols_has_ans[0, k] == cols_has_ans[0, answer_start_idx]:
                                        answer_stop_idx = k
                                    else:
                                        break

                                for k in reversed(list(range(0, answer_stop_idx2 - 1))):
                                    if cols_has_ans[0, k] == cols_has_ans[0, answer_stop_idx2]:
                                        answer_start_idx2 = k
                                    else:
                                        break

                                prob_1 = probs_start[0, answer_start_idx] + \
                                         probs_stop[0, answer_stop_idx]

                                prob_2 = probs_start[0, answer_start_idx2] + \
                                         probs_stop[0, answer_stop_idx2]

                                if prob_2 > prob_1:
                                    answer_start_idx = answer_start_idx2
                                    answer_stop_idx = answer_stop_idx2

                            score = probs_start[j, answer_start_idx]
                            scores.append(score * 1)
                            candi_scores.append(score * 1)

                            if answer_start_idx > answer_stop_idx:
                                answer_stop_idx = answer_start_idx + 15
                            if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                for k in range(answer_start_idx, input_ids.shape[1]):
                                    if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                        answer_stop_idx = k
                                        break

                            answer = ''

                            if answer_stop_idx + 1 >= input_ids.shape[1]:
                                answer_stop_idx = input_ids.shape[1] - 2

                            probs1 = probs_start[0, answer_start_idx]
                            probs2 = probs_stop[0, answer_stop_idx]
                            new_start_idx = answer_start_idx
                            new_stop_idx = answer_stop_idx

                            for k in range(answer_start_idx, answer_stop_idx + 1):
                                if k < 512 - 1:
                                    if cols_has_ans[0, k] != cols_has_ans[0, k + 1]:
                                        probs1 += probs_stop[0, k]
                                        new_stop_idx = k
                                        break

                            for k in list(reversed(range(answer_start_idx, answer_stop_idx + 1))):
                                if k > 1:
                                    if cols_has_ans[0, k] != cols_has_ans[0, k - 1]:
                                        probs2 += probs_start[0, k]
                                        new_start_idx = k
                                        break

                            if probs1 > probs2:
                                answer_stop_idx = new_stop_idx
                            else:
                                answer_start_idx = new_start_idx

                            for a_i, k in enumerate(range(answer_start_idx, answer_stop_idx + 1)):
                                if a_i > 1:
                                    if cols_has_ans[0, k] != cols_has_ans[0, k - 1]:
                                        break
                                answer += str(f_tokenizer.inv_vocab[input_ids[0, k]])

                            answers.append(answer.replace(' ##', ''))

                    if len(answers) > 0:
                        answer_candidates = []
                        candidates_scores = []

                        for _ in range(1):
                            m_s = -99
                            m_ix = 0

                            for q in range(len(scores)):
                                if m_s < scores[q]:
                                    m_s = scores[q]
                                    m_ix = q

                            answer_candidates.append(answer_re_touch(answers[m_ix]))
                            candidates_scores.append(candi_scores[m_ix])
                            # print('score:', scores[m_ix])
                            # scores[m_ix] = -999

                        a1 = [0]
                        a2 = [0]

                        for a_c in answer_candidates:
                            if a_c.find('<table>') != -1:
                                continue

                            a1.append(
                                exact_match_score(prediction=a_c, ground_truth=answer_text))
                            a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                        em_total += max(a1)
                        f1_total += max(a2)
                        epo += 1

                        for j in range(input_ids.shape[0]):
                            check = 'None'
                            answer_text = answer_text.replace('<a>', '').replace('</a>', '')

                            f1_ = f1_score(prediction=answer_re_touch(answers[j]),
                                           ground_truth=answer_text)

                            text = answers[j]

                            print('score:', scores[j], check, type, 'F1:',
                                  f1_, ' , ',
                                  text.replace('\n', ' '))
                        # print(table_text.replace('\n', ''))
                        # print('question:', question)
                        print('answer:', answer_text)
                        print('EM:', em_total / epo)
                        print('F1:', f1_total / epo)
                        print('-----\n', epo)

    def rename(self, replace_from, replace_to, add_prefix, dry_run):
        checkpoint_dir = self.save_path

        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        with tf.Session() as sess:
            for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
                # Load the variable
                var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

                # Set the new name
                new_name = var_name
                if None not in [replace_from, replace_to]:
                    new_name = new_name.replace(replace_from, replace_to).replace('MRC_block',
                                                                                  'table_mrc_block') \
                        .replace('pointer_net1', 'table_mrc_pointer_net')

                if add_prefix:
                    new_name = add_prefix + new_name

                if dry_run:
                    print('%s would be renamed to %s.' % (var_name, new_name))
                else:
                    print('Renaming %s to %s.' % (var_name, new_name))
                    # Rename the variable
                    var = tf.Variable(var, name=new_name)

            save_point = 'E:\\tapas_table_model\\qa_model.ckpt'

            if not dry_run:
                # Save the variables
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                saver.save(sess, save_point)

    def eval_with_span_test(self):
        dataholder = DataHolder_test.DataHolder()

        vocab = tokenization.load_vocab(vocab_file='vocab.txt')
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        f_tokenizer = tokenization.FullTokenizer(vocab_file='vocab.txt')

        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        tokenizer.add_tokens('[table]')
        tokenizer.add_tokens('[/table]')
        tokenizer.add_tokens('[list]')
        tokenizer.add_tokens('[/list]')
        tokenizer.add_tokens('[h3]')
        tokenizer.add_tokens('[td]')

        em_total = 0
        f1_total = 0
        epo = 0

        with tf.Session(config=config) as sess:
            model, bert_variables, sequence_output = self.create_model(self.input_ids, self.input_mask,
                                                                       self.input_segments, is_training=False)

            input_shape = get_shape_list(sequence_output, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]

            column_memory, row_memory = self.Table_Memory_Network(sequence_output=sequence_output, hops=3, dropout=0.0)

            row_one_hot = tf.one_hot(self.input_rows, depth=100)
            column_one_hot = tf.one_hot(self.input_cols, depth=50)

            column_memory = tf.matmul(column_one_hot, column_memory)
            row_memory = tf.matmul(row_one_hot, row_memory)

            sequence_output = tf.concat([column_memory, row_memory, sequence_output], axis=2)

            with tf.variable_scope("table_memory_hidden"):
                sequence_output = Fully_Connected(sequence_output, output=768, name='column_wise', activation=None)

            # sequence_output = tf.concat([names_embeddings, sequence_output], axis=2)
            prob_start, prob_stop = self.get_qa_probs(sequence_output, is_training=False)

            prob_start = tf.nn.softmax(prob_start, axis=-1)
            prob_stop = tf.nn.softmax(prob_stop, axis=-1)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, self.save_path)

            for _ in range(237):
                input_ids, input_mask, input_segments, input_rows, input_cols, \
                input_numeric_space, input_numeric_mask, answer_text, question_text = dataholder.next_batch()

                feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask,
                             self.input_segments: input_segments,
                             self.input_rows: input_rows, self.input_cols: input_cols,
                             }

                if input_ids.shape[0] > 0:

                    probs_start, probs_stop = \
                        sess.run([prob_start, prob_stop], feed_dict=feed_dict)

                    probs_start = np.array(probs_start, dtype=np.float32)
                    probs_stop = np.array(probs_stop, dtype=np.float32)

                    for j in range(input_ids.shape[0]):
                        for k in range(1, input_ids.shape[1]):
                            probs_start[j, k] = 0
                            probs_stop[j, k] = 0

                            if input_ids[j, k] == 3:
                                break

                    self.chuncker.get_feautre(answer_text)

                    prob_scores = []
                    c_scores = []

                    for j in range(input_ids.shape[0]):
                        # paragraph ranking을 위한 score 산정기준
                        # score2 = ev_values[j, 0]
                        score2 = 2 - (probs_start[j, 0] + probs_stop[j, 0])

                        prob_scores.append(score2)
                        # c_scores.append(self.chuncker.get_chunk_score(sequences[j]))

                    if True:
                        for j in range(input_ids.shape[0]):
                            probs_start[j, 0] = -999
                            probs_stop[j, 0] = -999

                        # CLS 선택 무효화

                        prediction_start = probs_start.argmax(axis=1)
                        prediction_stop = probs_stop.argmax(axis=1)

                        answers = []
                        scores = []
                        candi_scores = []

                        for j in range(input_ids.shape[0]):
                            answer_start_idx = prediction_start[j]
                            answer_stop_idx = prediction_stop[j]

                            if input_cols[0, answer_start_idx] != input_cols[0, answer_stop_idx]:
                                answer_stop_idx2 = answer_stop_idx
                                answer_stop_idx = answer_start_idx
                                answer_start_idx2 = answer_stop_idx2

                                for k in range(answer_start_idx + 1, input_ids.shape[1]):
                                    if input_cols[0, k] == input_cols[0, answer_start_idx]:
                                        answer_stop_idx = k
                                    else:
                                        break

                                for k in reversed(list(range(0, answer_stop_idx2 - 1))):
                                    if input_cols[0, k] == input_cols[0, answer_stop_idx2]:
                                        answer_start_idx2 = k
                                    else:
                                        break

                                prob_1 = probs_start[0, answer_start_idx] + \
                                         probs_stop[0, answer_stop_idx]

                                prob_2 = probs_start[0, answer_start_idx2] + \
                                         probs_stop[0, answer_stop_idx2]

                                if prob_2 > prob_1:
                                    answer_start_idx = answer_start_idx2
                                    answer_stop_idx = answer_stop_idx2

                            score = probs_start[j, answer_start_idx]
                            scores.append(score * 1)
                            candi_scores.append(score * 1)

                            if answer_start_idx > answer_stop_idx:
                                answer_stop_idx = answer_start_idx + 15
                            if f_tokenizer.inv_vocab[input_ids[j, answer_start_idx]] == '[p]':
                                for k in range(answer_start_idx, input_ids.shape[1]):
                                    if f_tokenizer.inv_vocab[input_ids[j, k]] == '[/p]':
                                        answer_stop_idx = k
                                        break

                            answer = ''

                            if answer_stop_idx + 1 >= input_ids.shape[1]:
                                answer_stop_idx = input_ids.shape[1] - 2

                            for k in range(answer_start_idx, answer_stop_idx + 1):
                                tok = f_tokenizer.inv_vocab[input_ids[j, k]]
                                if len(tok) > 0:
                                    if tok[0] != '#':
                                        answer += ' '
                                answer += str(f_tokenizer.inv_vocab[input_ids[j, k]]).replace(
                                    '##', '')

                            answers.append(answer)

                    if len(answers) > 0:
                        answer_candidates = []
                        candidates_scores = []

                        for _ in range(1):
                            m_s = -99
                            m_ix = 0

                            for q in range(len(scores)):
                                if m_s < scores[q]:
                                    m_s = scores[q]
                                    m_ix = q

                            answer_candidates.append(answer_re_touch(answers[m_ix]))
                            candidates_scores.append(candi_scores[m_ix])
                            # print('score:', scores[m_ix])
                            # scores[m_ix] = -999

                        a1 = [0]
                        a2 = [0]

                        for a_c in answer_candidates:
                            if a_c.find('<table>') != -1:
                                continue

                            a1.append(
                                exact_match_score(prediction=a_c, ground_truth=answer_text))
                            a2.append(f1_score(prediction=a_c, ground_truth=answer_text))

                        em_total += max(a1)
                        if max(a1) > 0.99:
                            a2.append(1.0)
                        f1_total += max(a2)
                        epo += 1

                        for j in range(input_ids.shape[0]):
                            check = 'None'
                            answer_text = answer_text.replace('<a>', '').replace('</a>', '')

                            f1_ = f1_score(prediction=answer_re_touch(answers[j]),
                                           ground_truth=answer_text)

                            text = answers[j]

                        print('score:', scores[j], check, type, 'F1:',
                              f1_, ' , ',
                              text.replace('\n', ' '))
                        # print(table_text.replace('\n', ''))
                        print('question:', question_text)
                        print('answer:', answer_text, ',')
                        print('EM:', em_total / epo)
                        print('F1:', f1_total / epo)
                        print('-----\n', epo)

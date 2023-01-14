import numpy as np
import tokenization

f_tk = tokenization.FullTokenizer(vocab_file='html_mecab_vocab_128000.txt')
inv_vocab = f_tk.inv_vocab


class DataHolder:
    def __init__(self):
        self.answer_span_list = np.load('answer_text_array_factory.npy')
        self.question_span_list = np.load('question_text_array_factory.npy')

        self.input_ids_table = np.load('sequence_table_factory.npy')
        self.input_mask_table = np.load('mask_table_factory.npy')
        self.input_segments_table = np.load('segments_table_factory.npy')
        self.input_positions_table = np.load('segments_table.npy')
        self.input_rows_table = np.load('rows_table_factory.npy')
        self.input_cols_table = np.load('cols_table_factory.npy')
        self.input_numeric_space = np.load('numeric_space.npy')
        self.input_numeric_mask = np.load('numeric_mask.npy')
        self.answer_span_table = np.load('answer_span_table.npy')

        self.r_ix = np.array(range(self.input_ids_table.shape[0]), dtype=np.int32)
        #np.random.shuffle(self.r_ix)


        self.batch_size = 1
        self.b_ix = 0
        self.b_ix2 = 0

        self.step = 0

        print(self.input_ids_table.shape)

    def next_batch(self):
        if self.b_ix + self.batch_size > self.input_ids_table.shape[0]:
            self.b_ix = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids_table.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids_table.shape[1]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids_table.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids_table.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids_table.shape[1]], dtype=np.int32)
        input_numeric_space = np.zeros(shape=[self.batch_size, 10, self.input_ids_table.shape[1]], dtype=np.int32)
        input_numeric_mask = np.zeros(shape=[self.batch_size, 10, self.input_ids_table.shape[1]], dtype=np.int32)
        start_label = np.zeros(shape=[self.batch_size, self.input_ids_table.shape[1]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids_table.shape[1]], dtype=np.int32)

        for i in range(1):
            ix = self.b_ix

            try:
                answer_text = self.answer_span_list[ix]
                question_text = self.question_span_list[ix]

            except:
                self.b_ix += 1
                return self.next_batch()

            input_ids[i] = self.input_ids_table[ix]
            input_segments[i] = self.input_segments_table[ix]
            input_mask[i] = self.input_mask_table[ix]
            input_rows[i] = self.input_rows_table[ix]
            input_cols[i] = self.input_cols_table[ix]

            self.b_ix += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, \
               input_numeric_space, input_numeric_mask, answer_text, question_text

"""
f_tokenizer = tokenization.FullTokenizer(vocab_file='html_mecab_vocab_128000.txt')
inv = f_tokenizer.inv_vocab

holder = DataHolder()

for i in range(10000):
    start_idx = holder.answer_span[i, 0]
    stop_idx = holder.answer_span[i, 1]

    cols = []
    rows = []

    word = ''
    word2 = ''

    for j in range(start_idx, stop_idx + 1):
        word2 += inv[holder.input_ids[i, j]] + ' '

    for j in range(0, stop_idx + 5):
        word += inv[holder.input_ids[i, j]] + ' '
        cols.append(holder.input_cols[i, j])
        rows.append(holder.input_rows[i, j])

    print(word)
    print(word2.replace(' ##', ''))
    print(cols)
    print(rows)


"""
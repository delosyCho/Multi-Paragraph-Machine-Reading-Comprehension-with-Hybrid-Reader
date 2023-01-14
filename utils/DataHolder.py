import numpy as np

class DataHolder:
    def __init__(self):
        self.input_ids_f = np.load('input_ids_finetuning.npy')
        self.input_segments_f = np.load('input_segments_finetuning.npy')
        self.answer_span_f = np.load('answer_span_finetuning.npy')

        self.input_ids = np.load('sequence_table.npy')
        self.input_segments = np.load('segments_table.npy')
        self.input_rows = np.load('rows_table.npy')
        self.input_cols = np.load('cols_table.npy')
        self.answer_span = np.load('answer_span_table.npy')

        self.input_text = np.load('sequence_crs2.npy')
        self.segments_text = np.load('segments_crs2.npy')
        self.answer_span_text = np.load('answer_span_crs2.npy')
        #self.additional_matrix = np.load('pos_attention_matrix.npy')
        print(self.input_text.shape, self.input_ids.shape)

        self.input_k1 = np.load('input_ids.npy')
        self.segments_k1 = np.load('segment_ids.npy')
        self.answer_span_k1 = np.load('answer_span.npy')

        self.r_ix = np.array(range(self.input_ids.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix)

        self.r_ix2 = np.array(range(self.input_text.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix2)

        self.r_ix3 = np.array(range(self.input_k1.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix3)

        self.r_ix4 = np.array(range(self.input_ids.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix4)
        """
        self.r_ix4 = np.array(range(self.input_rank.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix4)

        self.r_ix1 = np.array(range(self.input_k1.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix1)
        """
        self.batch_size = 4
        self.b_ix = 0
        self.b_ix1 = 0
        self.b_ix2 = 0
        self.b_ix3 = 0
        self.b_ix4 = 0

        self.step = 0

    def next_batch_z(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_label = np.zeros(shape=[self.batch_size, 2], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        for i in range(int(self.batch_size / 2)):
            rank_weights[i] = 1
            rank_label[i, 0] = 1
            rank_label[i + int(self.batch_size / 2), 1] = 1
            print(i, i + int(self.batch_size / 2))

            if self.step % 4 == 0:
                input_weights[i] = 1

                # table case
                while True:
                    ix = self.r_ix[self.b_ix]
                    try:
                        start_label[i, self.answer_span[ix, 0]] = 1
                        stop_label[i, self.answer_span[ix, 1]] = 1
                        break
                    except:
                        self.b_ix += 1

                input_ids[i] = self.input_ids[ix, 0]
                input_segments[i] = self.input_segments[ix, 0]
                input_rows[i] = self.input_rows[ix, 0]
                input_cols[i] = self.input_cols[ix, 0]

                input_ids[i + int(self.batch_size / 2)] = self.input_ids[ix, 1]
                input_segments[i + int(self.batch_size / 2)] = self.input_segments[ix, 1]
                input_rows[i + int(self.batch_size / 2)] = self.input_rows[ix, 1]
                input_cols[i + int(self.batch_size / 2)] = self.input_cols[ix, 1]
                self.b_ix += 1
            else:
                # table case
                while True:
                    ix = self.r_ix2[self.b_ix2]
                    try:
                        start_label[i, self.answer_span_text[ix, 0]] = 1
                        stop_label[i, self.answer_span_text[ix, 1]] = 1
                        break
                    except:
                        self.b_ix2 += 1

                input_ids[i] = self.input_text[ix, 0]
                input_segments[i] = self.segments_text[ix, 0]

                input_ids[i + int(self.batch_size / 2)] = self.input_text[ix, 1]
                input_segments[i + int(self.batch_size / 2)] = self.segments_text[ix, 1]
                self.b_ix2 += 1
            self.step += 1

        return input_ids, input_segments, start_label, stop_label, \
               input_rows, input_cols, rank_label, rank_weights, input_weights

    def next_batch_w(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        if self.b_ix3 + self.batch_size * 2 > self.input_k1.shape[0]:
            self.b_ix3 = 0

        if self.b_ix4 + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix4 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_label = np.zeros(shape=[self.batch_size, 2], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        i = 0
        input_weights[i] = 1
        rank_label[i, 0] = 1
        ix = self.r_ix2[self.b_ix2]
        try:
            start_label[i, self.answer_span_text[ix, 0]] = 1
            stop_label[i, self.answer_span_text[ix, 1]] = 1
        except:
            self.b_ix2 += 1
            self.next_batch_w()

        input_ids[i] = self.input_text[ix, 0]
        input_segments[i] = self.segments_text[ix, 0]

        input_weights[i + 1] = 0
        rank_label[i + 1, 1] = 1
        input_ids[i + 1] = self.input_text[ix, 1]
        input_segments[i + 1] = self.segments_text[ix, 1]
        start_label[i + 1, 0] = 1
        stop_label[i + 1, 0] = 1
        self.b_ix2 += 1

        if self.step % 2 == 0:
            i = 2
            input_weights[i] = 1
            rank_label[i, 0] = 1
            ix = self.r_ix4[self.b_ix4]
            try:
                start_label[i, self.answer_span_text[ix, 0]] = 1
                stop_label[i, self.answer_span_text[ix, 1]] = 1
            except:
                self.b_ix4 += 1
                self.next_batch_w()
            input_ids[i] = self.input_text[ix, 0]
            input_segments[i] = self.segments_text[ix, 0]
            self.b_ix4 += 1

            i = 3
            input_weights[i] = 1
            rank_label[i, 0] = 1
            ix = self.r_ix3[self.b_ix3]
            try:
                start_label[i, self.answer_span_k1[ix, 0]] = 1
                stop_label[i, self.answer_span_k1[ix, 1]] = 1
            except:
                self.b_ix3 += 1
                self.next_batch_w()
            input_ids[i] = self.input_k1[ix]
            input_segments[i] = self.segments_k1[ix]
            self.b_ix3 += 1
        else:
            i = 2
            input_weights[i] = 1
            rank_label[i, 0] = 1
            ix = self.r_ix[self.b_ix]

            try:
                start_label[i, self.answer_span[ix, 0]] = 1
                stop_label[i, self.answer_span[ix, 1]] = 1
            except:
                self.b_ix += 1
                return self.next_batch_w()

            input_ids[i] = self.input_ids[ix, 0]
            input_segments[i] = self.input_segments[ix, 0]
            input_rows[i] = self.input_rows[ix, 0]
            input_cols[i] = self.input_cols[ix, 0]
            self.b_ix += 1

            i = 3
            input_weights[i] = 0
            rank_label[i, 1] = 1
            input_ids[i] = self.input_ids[ix, 1]
            input_segments[i] = self.input_segments[ix, 1]
            input_rows[i] = self.input_rows[ix, 1]
            input_cols[i] = self.input_cols[ix, 1]
            start_label[i, 0] = 1
            stop_label[i, 0] = 1
        self.step += 1

        return input_ids, input_mask, input_segments, start_label, stop_label, input_weights, \
               input_rows, input_cols, rank_label

    def next_batch_f(self):
        if self.b_ix4 + self.batch_size * 2 > self.input_ids_f.shape[0]:
            self.b_ix4 = 0

        length = 512
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        for i in range(4):
            ix = self.r_ix4[self.b_ix4]
            try:
                start_label[i, self.answer_span_text[ix, 0]] = 1
                stop_label[i, self.answer_span_text[ix, 1]] = 1
            except:
                self.b_ix4 += 1
                self.next_batch_f()

            input_ids[i] = self.input_text[ix, 0]
            input_segments[i] = self.segments_text[ix, 0]
            self.b_ix4 += 1

        return input_ids, input_segments, start_label, stop_label

    def next_batch(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_label = np.zeros(shape=[self.batch_size, 2], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        start_label[-1, 0] = 1
        stop_label[-1, 0] = 1
        rank_label[-1, 1] = 1
        rank_weights[-1] = 0
        input_weights[-1] = 0

        check = False
        for i in range(2):
            input_weights[i] = 0
            rank_weights[i] = 1
            rank_label[i, 0] = 1

            ix = self.r_ix2[self.b_ix2]
            try:
                start_label[i, self.answer_span_text[ix, 0]] = 1
                stop_label[i, self.answer_span_text[ix, 1]] = 1
            except:
                self.b_ix2 += 1
                self.next_batch()

            input_ids[i] = self.input_text[ix, 0]
            input_segments[i] = self.segments_text[ix, 0]
            self.b_ix2 += 1
            #"""
            if self.step % 3 != 0 and check is False:
                if self.answer_span_text[ix, 2] == 1:
                    check = True
                    input_ids[-1] = self.input_text[ix, 1]
                    input_segments[-1] = self.segments_text[ix, 1]
            #"""
        #"""
        for i in range(2, 3):
            input_weights[i] = 1
            rank_weights[i] = 1
            rank_label[i, 0] = 1

            ix = self.r_ix[self.b_ix]
            try:
                start_label[i, self.answer_span[ix, 0]] = 1
                stop_label[i, self.answer_span[ix, 1]] = 1
            except:
                self.b_ix += 1
                return self.next_batch()

            input_ids[i] = self.input_ids[ix, 0]
            input_segments[i] = self.input_segments[ix, 0]
            input_rows[i] = self.input_rows[ix, 0]
            input_cols[i] = self.input_cols[ix, 0]
            self.b_ix += 1

            if self.step % 3 == 0 or check is False:
                input_ids[-1] = self.input_ids[ix, 1]
                input_segments[-1] = self.input_segments[ix, 1]
                input_rows[-1] = self.input_rows[ix, 1]
                input_cols[-1] = self.input_cols[ix, 1]
        #"""
        self.step += 1

        return input_ids, input_mask, input_segments, start_label, stop_label, input_weights, \
               input_rows, input_cols, rank_label, rank_weights

    def next_batch_text_only(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_label = np.zeros(shape=[self.batch_size, 2], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        start_label[-1, 0] = 1
        stop_label[-1, 0] = 1
        rank_label[-1, 1] = 1
        rank_weights[-1] = 0
        input_weights[-1] = 0

        check = False
        for i in range(2):
            input_weights[i] = 1
            rank_weights[i] = 1
            rank_label[i, 0] = 1

            ix = self.r_ix2[self.b_ix2]
            try:
                start_label[i, self.answer_span_text[ix, 0]] = 1
                stop_label[i, self.answer_span_text[ix, 1]] = 1
            except:
                self.b_ix2 += 1
                self.next_batch()

            input_ids[i] = self.input_text[ix, 0]
            input_segments[i] = self.segments_text[ix, 0]
            self.b_ix2 += 1

            input_ids[i + 2] = self.input_text[ix, 1]
            input_segments[i + 2] = self.segments_text[ix, 1]

        self.step += 1

        return input_ids, input_mask, input_segments, start_label, stop_label, input_weights, \
               input_rows, input_cols, rank_label, rank_weights


    def next_batch_3_batch(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_label = np.zeros(shape=[self.batch_size, 2], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        start_label[-1, 0] = 1
        stop_label[-1, 0] = 1
        rank_label[-1, 1] = 1
        rank_weights[-1] = 0
        input_weights[-1] = 0

        check = False
        for i in range(2):
            self.step += 1

            if self.step % 4 == 0:
                input_weights[i] = 1
                rank_weights[i] = 1
                rank_label[i, 0] = 1

                ix = self.r_ix[self.b_ix]
                try:
                    start_label[i, self.answer_span[ix, 0]] = 1
                    stop_label[i, self.answer_span[ix, 1]] = 1
                except:
                    self.b_ix += 1
                    return self.next_batch()

                input_ids[i] = self.input_ids[ix, 0]
                input_segments[i] = self.input_segments[ix, 0]
                input_rows[i] = self.input_rows[ix, 0]
                input_cols[i] = self.input_cols[ix, 0]
                self.b_ix += 1

                input_ids[-1] = self.input_ids[ix, 1]
                input_segments[-1] = self.input_segments[ix, 1]
                input_rows[-1] = self.input_rows[ix, 1]
                input_cols[-1] = self.input_cols[ix, 1]
                check = True
            else:
                input_weights[i] = 0
                rank_weights[i] = 1
                rank_label[i, 0] = 1

                ix = self.r_ix2[self.b_ix2]
                try:
                    start_label[i, self.answer_span_text[ix, 0]] = 1
                    stop_label[i, self.answer_span_text[ix, 1]] = 1
                except:
                    self.b_ix2 += 1
                    self.next_batch()

                input_ids[i] = self.input_text[ix, 0]
                input_segments[i] = self.segments_text[ix, 0]
                self.b_ix2 += 1
                #"""
                if self.step % 3 != 0 and check is False:
                    if self.answer_span_text[ix, 2] == 1:
                        check = True
                        input_ids[-1] = self.input_text[ix, 1]
                        input_segments[-1] = self.segments_text[ix, 1]
            #"""
        #"""
        #"""

        return input_ids, input_mask, input_segments, start_label, stop_label, input_weights, \
               input_rows, input_cols, rank_label, rank_weights

    def next_batch_table_only(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        rank_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        rank_label = np.zeros(shape=[self.batch_size, 2], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        start_label[-1, 0] = 1
        stop_label[-1, 0] = 1
        rank_label[-1, 1] = 1
        rank_weights[-1] = 0

        for i in range(2):
            rank_weights[i] = 1
            rank_label[i, 0] = 1

            ix = self.r_ix[self.b_ix]
            try:
                start_label[i, self.answer_span[ix, 0]] = 1
                stop_label[i, self.answer_span[ix, 1]] = 1
            except:
                self.b_ix += 1
                return self.next_batch_table_only()

            input_ids[i] = self.input_ids[ix, 0]
            input_segments[i] = self.input_segments[ix, 0]
            input_rows[i] = self.input_rows[ix, 0]
            input_cols[i] = self.input_cols[ix, 0]

            if i + 2 < self.batch_size:
                rank_weights[i + 2] = 0
                rank_label[i + 2, 1] = 1
                input_ids[i + 2] = self.input_ids[ix, 1]
                input_segments[i + 2] = self.input_segments[ix, 1]
                input_rows[i + 2] = self.input_rows[ix, 1]
                input_cols[i + 2] = self.input_cols[ix, 1]
            self.b_ix += 1

        return input_ids, input_segments, start_label, stop_label, \
               input_rows, input_cols, rank_label, rank_weights

    def next_batch_(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        for i in range(4):
            input_weights[i] = 0
            ix = self.r_ix2[self.b_ix2]
            try:
                start_label[i, self.answer_span_text[ix, 0]] = 1
                stop_label[i, self.answer_span_text[ix, 1]] = 1
            except:
                self.b_ix2 += 1
                self.next_batch()

            input_ids[i] = self.input_text[ix]
            input_segments[i] = self.segments_text[ix]
            self.b_ix2 += 1
        self.step += 1

        return input_ids, input_mask, input_segments, start_label, stop_label, input_weights, \
               input_rows, input_cols

    def next_batch_k1(self):
        if self.b_ix + self.batch_size * 2 > self.input_ids.shape[0]:
            self.b_ix = 0

        if self.b_ix2 + self.batch_size * 2 > self.input_text.shape[0]:
            self.b_ix2 = 0

        if self.b_ix3 + self.batch_size * 2 > self.input_k1.shape[0]:
            self.b_ix3 = 0

        length = self.input_ids.shape[2]
        input_ids = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)
        pos_attention_matrix = np.zeros(shape=[self.batch_size, length, length], dtype=np.float32)

        start_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, length], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix3[self.b_ix3]

            input_weights[i] = 0
            try:
                start_label[i, self.answer_span_k1[ix, 0]] = 1
                stop_label[i, self.answer_span_k1[ix, 1]] = 1
            except:
                self.b_ix3 += 1
                return self.next_batch_k1()

            input_ids[i] = self.input_k1[ix]
            input_segments[i] = self.segments_k1[ix]
            pos_attention_matrix[i] = self.additional_matrix[ix]

            self.b_ix3 += 1

        return input_ids, input_mask, input_segments, start_label, stop_label, input_weights, \
               input_rows, input_cols, pos_attention_matrix

    def next_batch_korquad1(self):
        if self.b_ix3 + self.batch_size >= self.input_k1.shape[0]:
            np.random.shuffle(self.r_ix3)
            self.b_ix3 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix3[self.b_ix3]

            input_ids[i] = self.input_k1[ix]
            input_segments[i] = self.segments_k1[ix]

            start_label[i, self.answer_span_k1[ix, 0]] = 1
            stop_label[i, self.answer_span_k1[ix, 1]] = 1

            self.b_ix3 += 1

        return input_ids, input_segments, start_label, stop_label

    def next_batch_book(self):
        if self.b_ix1 + self.batch_size >= self.input_b.shape[0]:
            np.random.shuffle(self.r_ix3)
            self.b_ix1 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix1[self.b_ix1]

            input_ids[i] = self.input_b[ix]
            input_segments[i] = self.segments_b[ix]

            start_label[i, self.answer_span_b[ix, 0]] = 1
            stop_label[i, self.answer_span_b[ix, 1]] = 1

            self.b_ix1 += 1

        return input_ids, input_segments, start_label, stop_label
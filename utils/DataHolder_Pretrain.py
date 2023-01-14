import numpy as np
import random


class DataHolder:
    def __init__(self):
        self.input_ids2 = np.load('sequence_table2.npy')
        self.input_mask2 = np.load('mask_table2.npy')
        self.input_segments2 = np.load('segments_table2.npy')
        self.input_rows2 = np.load('rows_table2.npy')
        self.input_cols2 = np.load('cols_table2.npy')
        self.answer_span2 = np.load('answer_span_table2.npy')

        self.input_ids_te = np.load('all_ids.npy')
        self.input_ids_te = np.transpose(self.input_ids_te, [0, 2, 1])

        self.label_te = np.load('label.npy')

        input_ids1 = np.load('sequence_table_pre2.npy')
        label_ids1 = np.load('label_ids2.npy')

        self.input_ids1 = np.concatenate([input_ids1], axis=0)
        self.label_ids1 = np.concatenate([label_ids1], axis=0)

        self.input_ids_cs = np.load('input_ids_cs.npy')
        self.label_cs = np.load('answer_span_cs.npy')

        input_ids_rtp1 = np.load('input_ids_RTP.npy')
        input_ids_rtp2 = np.load('input_ids_RTP2.npy')
        self.input_ids_rtp = np.concatenate([input_ids_rtp1, input_ids_rtp2], axis=0)
        self.input_ids_se = np.load('input_ids_SE.npy')

        print('entailment ids:', self.input_ids_rtp.shape, self.input_ids_se.shape)

        self.r_ix_rtp = np.array(range(self.input_ids_rtp.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix_rtp)

        self.r_ix_se = np.array(range(self.input_ids_se.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix_se)

        self.r_ix1 = np.array(range(self.input_ids1.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix1)

        self.r_ix2 = np.array(range(self.input_ids_cs.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix2)

        self.r_ix3 = np.array(range(self.input_ids_te.shape[0]), dtype=np.int32)
        #np.random.shuffle(self.r_ix3)

        self.r_ix = np.array(range(self.input_ids2.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix)

        self.batch_size = 4
        self.b_ix1 = 0
        self.b_ix2 = 0
        self.b_ix3 = 0

        self.b_ix = 0

        self.step = 0

    def next_batch_entailment(self):
        if self.b_ix1 + self.batch_size > self.input_ids_rtp.shape[0]:
            self.b_ix1 = 0

        if self.b_ix2 + self.batch_size > self.input_ids_se.shape[0]:
            self.b_ix2 = 0

        seq_length = 512
        input_ids = np.zeros(shape=[self.batch_size, seq_length], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, seq_length], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, seq_length], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, seq_length], dtype=np.int32)

        input_weights = np.zeros(shape=[self.batch_size], dtype=np.float32)

        label_rtp = np.zeros(shape=[self.batch_size, 2], dtype=np.int32)
        label_se = np.zeros(shape=[self.batch_size, 2], dtype=np.int32)
        print('batch:', self.batch_size)
        for i in range(0, 1):
            input_weights[i] = 1
            input_weights[i + 2] = 1

            ix = self.r_ix_rtp[self.b_ix1]

            input_ids[i] = self.input_ids_rtp[ix, 0, 0]
            input_segments[i] = self.input_ids_rtp[ix, 0, 1]
            input_cols[i] = self.input_ids_rtp[ix, 0, 3]
            input_rows[i] = self.input_ids_rtp[ix, 0, 4]

            input_ids[i + 2] = self.input_ids_rtp[ix, 1, 0]
            input_segments[i + 2] = self.input_ids_rtp[ix, 1, 1]
            input_cols[i + 2] = self.input_ids_rtp[ix, 1, 3]
            input_rows[i + 2] = self.input_ids_rtp[ix, 1, 4]

            label_rtp[i, 0] = 1
            label_rtp[i + 2, 1] = 1
            self.b_ix1 += 1

        for i in range(1, 2):
            input_weights[i] = 0
            input_weights[i + 2] = 0

            ix = self.r_ix_se[self.b_ix2]

            input_ids[i] = self.input_ids_se[ix, 0, 0]
            input_segments[i] = self.input_ids_se[ix, 0, 1]
            input_cols[i] = self.input_ids_se[ix, 0, 3]
            input_rows[i] = self.input_ids_se[ix, 0, 4]

            input_ids[i + 2] = self.input_ids_se[ix, 1, 0]
            input_segments[i + 2] = self.input_ids_se[ix, 1, 1]
            input_cols[i + 2] = self.input_ids_se[ix, 1, 3]
            input_rows[i + 2] = self.input_ids_se[ix, 1, 4]

            label_se[i, 0] = 1
            label_se[i + 2, 1] = 1
            self.b_ix2 += 1

        return input_ids, input_segments, input_cols, input_rows, input_weights, label_rtp, label_se

    def next_batch(self):
        if self.b_ix1 + self.batch_size > self.input_ids1.shape[0]:
            self.b_ix1 = 0
        input_ids = np.zeros(shape=[self.batch_size, self.input_ids1.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids1.shape[2]], dtype=np.int32)
        input_mask = np.zeros(shape=[self.batch_size, self.input_ids1.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids1.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids1.shape[2]], dtype=np.int32)

        label_ids = np.zeros(shape=[self.batch_size, 20], dtype=np.int32)
        label_positions = np.zeros(shape=[self.batch_size, 20], dtype=np.int32)
        label_weight = np.zeros(shape=[self.batch_size, 20], dtype=np.float32)

        for i in range(self.batch_size):
            ix = self.r_ix1[self.b_ix1]

            input_ids[i] = self.input_ids1[ix, 0]
            input_segments[i] = self.input_ids1[ix, 1]
            input_mask[i] = self.input_ids1[ix, 2]
            input_cols[i] = self.input_ids1[ix, 3]
            input_rows[i] = self.input_ids1[ix, 4]

            label_ids[i] = self.label_ids1[ix, 0]
            label_positions[i] = self.label_ids1[ix, 1]
            label_weight[i] = self.label_ids1[ix, 2]

            self.b_ix1 += 1

        self.step += 1

        return input_ids, input_mask, input_segments, input_rows, input_cols, \
               label_ids, label_positions, label_weight

    def next_batch_cs(self):
        if self.b_ix2 + self.batch_size > self.input_ids_cs.shape[0]:
            self.b_ix2 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)

        label_rows = np.zeros(shape=[self.batch_size, 100], dtype=np.int32)
        label_cols = np.zeros(shape=[self.batch_size, 50], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix2[self.b_ix2]

            try:
                label_rows[i, self.label_cs[ix, 2]] = 1
                label_cols[i, self.label_cs[ix, 3]] = 1
            except:
                self.b_ix2 += 1
                return self.next_batch_cs()

            input_ids[i] = self.input_ids_cs[ix, 0]
            input_segments[i] = self.input_ids_cs[ix, 1]
            input_rows[i] = self.input_ids_cs[ix, 2]
            input_cols[i] = self.input_ids_cs[ix, 3]
            self.b_ix2 += 1

        self.step += 1

        return input_ids, input_segments, input_rows, input_cols, label_rows, label_cols

    def next_batch_cs2(self):
        if self.b_ix2 + self.batch_size > self.input_ids_cs.shape[0]:
            self.b_ix2 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)

        label_rows = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)
        label_cols = np.zeros(shape=[self.batch_size, self.input_ids_cs.shape[2]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix2[self.b_ix2]

            try:
                label_rows[i, self.label_cs[ix, 0]] = 1
                label_cols[i, self.label_cs[ix, 1]] = 1
            except:
                self.b_ix2 += 1
                return self.next_batch_cs2()

            input_ids[i] = self.input_ids_cs[ix, 0]
            input_segments[i] = self.input_ids_cs[ix, 1]
            input_rows[i] = self.input_ids_cs[ix, 2]
            input_cols[i] = self.input_ids_cs[ix, 3]

            self.b_ix2 += 1

        self.step += 1

        return input_ids, input_segments, input_rows, input_cols, label_rows, label_cols

    def next_batch_te(self):
        if self.b_ix3 + self.batch_size > self.input_ids_te.shape[0]:
            self.b_ix3 = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids_te.shape[2]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids_te.shape[2]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids_te.shape[2]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids_te.shape[2]], dtype=np.int32)

        label_entailment = np.zeros(shape=[self.batch_size, 2], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix3[self.b_ix3]

            input_ids[i] = self.input_ids_te[ix, 0]
            input_segments[i] = self.input_ids_te[ix, 1]
            input_rows[i] = self.input_ids_te[ix, 2]
            input_cols[i] = self.input_ids_te[ix, 3]

            label_entailment[i, self.label_te[ix]] = 1
            self.b_ix3 += 1

        self.step += 1

        return input_ids, input_segments, input_rows, input_cols, label_entailment

    def next_batch_tqa(self):
        if self.b_ix + self.batch_size > self.input_ids2.shape[0]:
            self.b_ix = 0

        input_ids = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_rows = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        input_cols = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        start_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)
        stop_label = np.zeros(shape=[self.batch_size, self.input_ids2.shape[1]], dtype=np.int32)

        for i in range(self.batch_size):
            ix = self.r_ix[self.b_ix]
            #print(self.b_ix, ix)
            try:
                start_label[i, self.answer_span2[ix, 0]] = 1
                stop_label[i, self.answer_span2[ix, 1]] = 1
            except:
                self.b_ix += 1
                return self.next_batch_tqa()

            input_ids[i] = self.input_ids2[ix]
            input_segments[i] = self.input_segments2[ix]
            input_rows[i] = self.input_rows2[ix]
            input_cols[i] = self.input_cols2[ix]
            self.b_ix += 1

        return input_ids, input_segments, input_rows, input_cols, start_label, stop_label

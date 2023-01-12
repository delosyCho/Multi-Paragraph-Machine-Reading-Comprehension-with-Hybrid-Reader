import math
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from modelings import gelu, Encoder

from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax, MSELoss
from transformers import AutoModel

from attention_utils import compute_sparse_attention


def Linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def make_attention_mask_3d(attention_mask):
    return None


class QuestionAnswering(nn.Module):
    def __init__(self):
        super(QuestionAnswering, self).__init__()
        #self.bert = AutoModel.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.bert = AutoModel.from_pretrained('monologg/kobigbird-bert-base')

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pool_hidden = nn.Linear(768, 768)

        self.qa_hidden1 = nn.Linear(768 * 4, 768 * 2)
        self.qa_hidden2 = nn.Linear(768 * 2, 768)

        self.qa_outputs = nn.Linear(768, 2)
        self.qa_outputs_table = nn.Linear(768, 2)

        self.vf_outputs = nn.Linear(768, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.dropout = Dropout(0.1)

        self.row_hidden = nn.Linear(768, 768)
        self.col_hidden = nn.Linear(768, 768)

        self.adapter_hidden1 = nn.Linear(768 * 3, 768)
        self.adapter_hidden2 = nn.Linear(768 * 3, 768)

        self.adapter_layers = Encoder(num_hidden_layers=1)
        self.adapter_layers2 = Encoder(num_hidden_layers=1)

    def forward(self,
                input_ids,
                token_type_ids=None, attention_mask=None,
                attention_mask_sparse=None,
                start_positions=None, end_positions=None,
                input_weights=None, sparse_attention_mask=None,
                pos_ids_tb=None,
                columns_ids_tb=None,
                columns_ids=None,
                rows_ids_tb=None,
                device=None,
                seq_length=2048):
        tb_size = 5
        tb_length = 512
        hidden_size = 768
        max_col = 512
        max_row = 512

        token_weights = torch.where(columns_ids > 0, torch.ones_like(input_ids),
                                    torch.zeros_like(input_ids))
        inverse_token_weights = torch.ones_like(token_weights) - token_weights

        sparse_attention_mask = (1.0 - sparse_attention_mask) * -10000.0
        sparse_attention_mask = torch.unsqueeze(sparse_attention_mask, dim=1)

        if start_positions is not None:
            start_positions = torch.reshape(start_positions, shape=[-1])
            end_positions = torch.reshape(end_positions, shape=[-1])

        if attention_mask is None:
            attention_mask = torch.where(input_ids > 0, torch.ones_like(input_ids), torch.zeros_like(input_ids))

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [B, S, H]

        pos_ids_tb = torch.reshape(pos_ids_tb, shape=[-1, tb_size * tb_length]) # [B, T * 512]
        pos_ids_tb_one_hot = one_hot(pos_ids_tb, num_classes=seq_length) # [B, T * 512, 2048]
        pos_ids_tb_one_hot = torch.tensor(pos_ids_tb_one_hot, dtype=torch.float)

        sequence_output_tb = torch.matmul(
            pos_ids_tb_one_hot, sequence_output
        )

        sequence_output_tb = torch.reshape(sequence_output_tb, shape=[-1, tb_length, 768]) # [B * T, 512, H]

        columns_ids_tb = torch.reshape(columns_ids_tb, shape=[-1, tb_length]) # [B * T, 512]
        columns_ids_tb_one_hot = one_hot(columns_ids_tb, num_classes=max_col).type(torch.float32) # [B * T, 512, C]
        columns_ids_tb_one_hot_ = torch.transpose(columns_ids_tb_one_hot, 1, 2) # [B * T, C, 512]

        rows_ids_tb = torch.reshape(rows_ids_tb, shape=[-1, tb_length]) # [B * T, 512]
        rows_ids_tb_one_hot = one_hot(rows_ids_tb, num_classes=max_col).type(torch.float32)  # [B * T, 512, R]
        rows_ids_tb_one_hot_ = torch.transpose(rows_ids_tb_one_hot, 1, 2)  # [B * T, R, 512]

        columns_hidden = torch.matmul(columns_ids_tb_one_hot_, sequence_output_tb)  # [B * T, C, H]
        columns_hidden = torch.matmul(columns_ids_tb_one_hot, columns_hidden)  # [B * T, C, H]

        rows_hidden = torch.matmul(rows_ids_tb_one_hot_, sequence_output_tb) # [B * T, R, H]
        rows_hidden = torch.matmul(rows_ids_tb_one_hot, rows_hidden)  # [B * T, C, H]

        sequence_output_tb = torch.cat([sequence_output_tb, columns_hidden, rows_hidden], dim=-1)
        sequence_output_tb = gelu(self.adapter_hidden1(sequence_output_tb))

        sparse_attention_mask = torch.reshape(sparse_attention_mask, shape=[-1, 1, tb_length, tb_length])
        sequence_output_tb = self.adapter_layers(
            hidden_states=sequence_output_tb,
            attention_mask=sparse_attention_mask
        )
        sequence_output_tb = sequence_output_tb[-1]
        #print('layer1:', torch.max(sequence_output_tb))

        columns_denominator = torch.matmul(columns_ids_tb_one_hot_, torch.ones_like(columns_hidden)) + 0.001
        columns_hidden = torch.matmul(columns_ids_tb_one_hot_, sequence_output_tb) / columns_denominator  # [B * T, C, H]
        columns_hidden = torch.matmul(columns_ids_tb_one_hot, columns_hidden)  # [B * T, C, H]

        rows_denominator = torch.matmul(rows_ids_tb_one_hot_, torch.ones_like(rows_hidden)) + 0.001
        rows_hidden = torch.matmul(rows_ids_tb_one_hot_, sequence_output_tb) / rows_denominator  # [B * T, R, H]
        rows_hidden = torch.matmul(rows_ids_tb_one_hot, rows_hidden)  # [B * T, C, H]

        sequence_output_tb = torch.cat([sequence_output_tb, columns_hidden, rows_hidden], dim=-1)
        sequence_output_tb = gelu(self.adapter_hidden2(sequence_output_tb))

        sequence_output_tb = self.adapter_layers2(
            hidden_states=sequence_output_tb,
            attention_mask=sparse_attention_mask
        )
        sequence_output_tb = sequence_output_tb[-1]
        #print('layer2:', torch.max(sequence_output_tb))

        columns_denominator = torch.matmul(columns_ids_tb_one_hot_, torch.ones_like(columns_hidden)) + 0.001
        columns_hidden = torch.matmul(columns_ids_tb_one_hot_, sequence_output_tb) / columns_denominator  # [B * T, C, H]
        columns_hidden = torch.matmul(columns_ids_tb_one_hot, columns_hidden)  # [B * T, C, H]

        #print('col:', torch.max(columns_hidden))
        rows_denominator = torch.matmul(rows_ids_tb_one_hot_, torch.ones_like(rows_hidden)) + 0.001
        rows_hidden = torch.matmul(rows_ids_tb_one_hot_, sequence_output_tb) / rows_denominator  # [B * T, R, H]
        rows_hidden = torch.matmul(rows_ids_tb_one_hot, rows_hidden)  # [B * T, C, H]

        #print('row:', torch.max(rows_hidden))
        sequence_output_tb = torch.cat([sequence_output_tb, columns_hidden, rows_hidden], dim=-1)
        sequence_output_tb = torch.reshape(sequence_output_tb, shape=[-1, tb_size * tb_length, hidden_size * 3])
        # [B, T * 512, H * 3]

        # [B, T * 512, 2048]
        pos_ids_tb_one_hot_ = torch.transpose(pos_ids_tb_one_hot, 1, 2)  # [B, 2048, T * 512]
        #print(input_ids.shape)
        #print(pos_ids_tb_one_hot_.shape, sequence_output_tb.shape)
        #print('check layer1:', torch.max(sequence_output_tb))
        batch_size = input_ids.shape[0]
        ones_denominator = torch.ones(size=[batch_size, tb_size * tb_length, 1], dtype=torch.float32)
        ones_denominator = ones_denominator.to(input_ids.device)

        sequence_table_denominator = torch.matmul(pos_ids_tb_one_hot_, ones_denominator) + 0.001
        sequence_table_output = torch.matmul(pos_ids_tb_one_hot_, sequence_output_tb) / sequence_table_denominator
        sequence_table_output = torch.cat([sequence_output, sequence_table_output], dim=-1) # [B, 2048, H * 4]
        #print(sequence_table_output.shape)
        # make predictions
        # predictions for texts
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        sequence_output_table_output = gelu(self.qa_hidden1(sequence_table_output))
        sequence_output_table_output = gelu(self.qa_hidden2(sequence_output_table_output))

        # predictions for tables
        logits = self.qa_outputs_table(sequence_output_table_output)
        start_logits2, end_logits2 = logits.split(1, dim=-1)
        start_logits2 = start_logits2.squeeze(-1).contiguous()
        end_logits2 = end_logits2.squeeze(-1).contiguous()

        #print(torch.max(start_logits), torch.max(start_logits2))
        #print()

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

            start_logits = start_logits * inverse_token_weights + start_logits2 * token_weights
            end_logits = end_logits * inverse_token_weights + end_logits2 * token_weights

            start_loss = torch.mean(loss_fct(start_logits, start_positions))
            end_loss = torch.mean(loss_fct(end_logits, end_positions))
            qa_loss = (start_loss + end_loss) / 2

            total_loss = qa_loss
            return total_loss, qa_loss
        else:
            start_logits = start_logits * inverse_token_weights + start_logits2 * token_weights
            end_logits = end_logits * inverse_token_weights + end_logits2 * token_weights

            start_logits = self.softmax(start_logits)
            end_logits = self.softmax(end_logits)

            return start_logits, end_logits

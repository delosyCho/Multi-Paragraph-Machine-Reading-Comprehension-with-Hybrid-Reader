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

        self.qa_hidden1 = nn.Linear(768 * 3, 768 * 2)
        self.qa_hidden2 = nn.Linear(768 * 2, 768)

        self.qa_outputs = nn.Linear(768, 2)
        self.qa_outputs_table = nn.Linear(768, 2)

        self.vf_outputs = nn.Linear(768, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.dropout = Dropout(0.1)

        self.row_hidden = nn.Linear(768, 768)
        self.col_hidden = nn.Linear(768, 768)

        self.transformers_row = Encoder(num_hidden_layers=2)
        self.transformers_col = Encoder(num_hidden_layers=2)

    def forward(self,
                input_ids,
                token_type_ids=None, attention_mask=None,
                columns_ids=None, rows_ids=None,
                attention_mask_cols=None, attention_mask_rows=None,
                start_positions=None, end_positions=None,
                input_weights=None):
        #print(columns_ids.shape, rows_ids.shape, input_ids.shape)
        token_weights = torch.where(columns_ids > 0, torch.ones_like(input_ids),
                                    torch.zeros_like(input_ids))
        inverse_token_weights = torch.ones_like(token_weights) - token_weights

        attention_mask_cols = (1.0 - attention_mask_cols) * -10000.0
        attention_mask_cols = torch.unsqueeze(attention_mask_cols, dim=1)

        attention_mask_rows = (1.0 - attention_mask_rows) * -10000.0
        attention_mask_rows = torch.unsqueeze(attention_mask_rows, dim=1)

        #print(attention_mask_rows.shape)
        #input()
        if start_positions is not None:
            start_positions = torch.reshape(start_positions, shape=[-1])
            end_positions = torch.reshape(end_positions, shape=[-1])

        if attention_mask is None:
            attention_mask = torch.where(input_ids > 0, torch.ones_like(input_ids), torch.zeros_like(input_ids))
        #print(input_ids.shape, token_type_ids.shape, attention_mask.shape)

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # [B, R, S] X [B, S, H] => [B, R, H]
        #print('attention row:', attention_mask_rows.shape)
        #print('row_ids:', rows_ids)
        row_one_hot = one_hot(rows_ids, num_classes=512).type(torch.float32)
        row_one_hot = torch.transpose(row_one_hot, 1, 2)

        row_hidden = torch.matmul(row_one_hot, sequence_output)
        row_hidden = gelu(self.row_hidden(row_hidden))
        #print(row_hidden.shape)
        #print(attention_mask_rows.shape)
        row_hidden_outputs = self.transformers_row(
            hidden_states=row_hidden,
            attention_mask=attention_mask_rows
        )
        row_hidden = row_hidden_outputs[-1]

        # [B, C, S] X [B, S, H] => [B, C, H]
        column_one_hot = one_hot(columns_ids, num_classes=512).type(torch.float32)
        column_one_hot = torch.transpose(column_one_hot, 1, 2)

        column_hidden = torch.matmul(column_one_hot, sequence_output)
        column_hidden = gelu(self.col_hidden(column_hidden))
        #print('column hidden:', column_hidden.shape)
        column_hidden_outputs = self.transformers_col(
            hidden_states=column_hidden,
            attention_mask=attention_mask_cols
        )
        column_hidden = column_hidden_outputs[-1]
        #print('column hidden2:', column_hidden.shape)

        # [B, S, R] X [B, R, H]
        row_one_hot = one_hot(rows_ids, num_classes=512).type(torch.float32)
        row_sequence_output = torch.matmul(row_one_hot, row_hidden)

        # [B, S, C] X [B, C, H]
        column_one_hot = one_hot(columns_ids, num_classes=512).type(torch.float32)
        column_sequence_output = torch.matmul(column_one_hot, column_hidden)

        # make predictions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        #print(row_one_hot.shape)
        #print(sequence_output.shape)
        #print(row_sequence_output.shape)
        #print(column_sequence_output.shape)
        sequence_output = torch.cat([sequence_output, row_sequence_output, column_sequence_output], dim=-1)
        sequence_output = gelu(self.qa_hidden1(sequence_output))
        sequence_output = gelu(self.qa_hidden2(sequence_output))

        logits = self.qa_outputs_table(sequence_output)
        start_logits2, end_logits2 = logits.split(1, dim=-1)
        start_logits2 = start_logits2.squeeze(-1).contiguous()
        end_logits2 = end_logits2.squeeze(-1).contiguous()

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

            #inverse_input_weights = torch.ones_like(input_weights) - input_weights
            #start_loss = torch.mean(loss_fct(start_logits, start_positions) * inverse_input_weights)
            #end_loss = torch.mean(loss_fct(end_logits, end_positions) * inverse_input_weights)
            #qa_loss = (start_loss + end_loss) / 2

            start_logits = start_logits * inverse_token_weights + start_logits2 * token_weights
            end_logits = end_logits * inverse_token_weights + end_logits2 * token_weights

            start_loss = torch.mean(loss_fct(start_logits, start_positions))
            end_loss = torch.mean(loss_fct(end_logits, end_positions))
            qa_loss2 = (start_loss + end_loss) / 2

            total_loss = qa_loss2
            return total_loss, qa_loss2
        else:
            inverse_input_weights = torch.ones_like(input_weights) - input_weights

            start_logits = self.softmax(start_logits) * inverse_input_weights
            end_logits = self.softmax(end_logits) * inverse_input_weights

            start_logits2 = self.softmax(start_logits2) * input_weights
            end_logits2 = self.softmax(end_logits2) * input_weights

            return start_logits + start_logits2, end_logits + end_logits2

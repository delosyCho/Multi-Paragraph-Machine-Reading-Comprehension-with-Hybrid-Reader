import math
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax, MSELoss
from transformers import AutoModel


def Linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class QuestionAnswering(nn.Module):
    def __init__(self):
        super(QuestionAnswering, self).__init__()
        #self.bert = AutoModel.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.bert = AutoModel.from_pretrained('monologg/kobigbird-bert-base')

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pool_hidden = nn.Linear(768, 768)

        self.qa_outputs = nn.Linear(768, 2)
        self.vf_outputs = nn.Linear(768, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.dropout = Dropout(0.1)

    def forward(self,
                input_ids,
                token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None,):

        if start_positions is not None:
            start_positions = torch.reshape(start_positions, shape=[-1])
            end_positions = torch.reshape(end_positions, shape=[-1])

        if attention_mask is None:
            attention_mask = torch.where(input_ids > 0, torch.ones_like(input_ids), torch.zeros_like(input_ids))
        #print(input_ids.shape, token_type_ids.shape, attention_mask.shape)

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

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

            start_loss = torch.mean(loss_fct(start_logits, start_positions))
            end_loss = torch.mean(loss_fct(end_logits, end_positions))
            qa_loss = (start_loss + end_loss) / 2

            total_loss = qa_loss
            return total_loss, qa_loss
        else:
            start_logits = self.softmax(start_logits)
            end_logits = self.softmax(end_logits)

            return start_logits, end_logits

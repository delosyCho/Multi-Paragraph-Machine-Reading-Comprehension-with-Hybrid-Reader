import math
import torch
import torch.nn as nn

import tag_op.tagop.attention_utils as attention_utils
# import attention_utils

from tag_op.tagop.modelings import gelu, Block
# from modelings import gelu, Encoder, Block

from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax, MSELoss
from torch.nn.functional import one_hot

from transformers import AutoModelForMaskedLM, AutoModel

LayerNorm = nn.LayerNorm


def Linear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class ProjectionLayer(nn.Module):
    def __init__(self, hidden_size=1024):
        super(ProjectionLayer, self).__init__()
        self.projection_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_sates, token_type_ids, max_length=256, with_cell_shape=False):
        one_hot_to = one_hot(token_type_ids, num_classes=max_length).type(torch.float32)
        one_hot_from = torch.transpose(one_hot_to, 1, 2)

        cell_hidden = torch.matmul(one_hot_from, hidden_sates)
        cell_hidden = gelu(self.projection_linear(cell_hidden))
        if with_cell_shape is True:
            return cell_hidden

        seq_hidden = torch.matmul(one_hot_to, cell_hidden)

        return seq_hidden


class TableWiseLayer(nn.Module):
    def __init__(self, hidden_size=1024, ff_dim=3072, dropout_prob=0.2):
        super(TableWiseLayer, self).__init__()
        self.down_proj_layer = Linear(hidden_size * 2, hidden_size)
        self.row_linear = nn.Linear(hidden_size, hidden_size)
        self.col_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = Dropout(dropout_prob)

    def forward(self, former_input, hidden_sates, token_type_ids, attention_mask, max_length=256):
        # print('former:', former_input.shape)
        # row_one_hot = one_hot(token_type_ids[:, :, 0], num_classes=512).type(torch.float32)
        # row_one_hot = torch.transpose(row_one_hot, 1, 2)
        sequence_output = gelu(self.down_proj_layer(torch.cat([former_input, hidden_sates], dim=-1)))

        one_hot_to = one_hot(token_type_ids, num_classes=max_length).type(torch.float32)
        one_hot_from = torch.transpose(one_hot_to, 1, 2)

        cell_hidden = torch.matmul(one_hot_from, sequence_output)
        cell_hidden = self.dropout(gelu(self.row_linear(cell_hidden)))

        seq_hidden = torch.matmul(one_hot_to, cell_hidden)

        return seq_hidden


class TableAdapterLayer(nn.Module):
    def __init__(self, hidden_size=1024, ff_dim=3072, dropout_prob=0.2):
        super(TableAdapterLayer, self).__init__()
        self.col_adapter_layer = TableWiseLayer(hidden_size=hidden_size, ff_dim=ff_dim, dropout_prob=dropout_prob)
        self.row_adapter_layer = TableWiseLayer(hidden_size=hidden_size, ff_dim=ff_dim, dropout_prob=dropout_prob)
        self.seq_adapter_layer = Block(hidden_size=hidden_size, ff_dim=ff_dim, dropout_prob=dropout_prob)

        self.down_proj_layer = Linear(hidden_size * 4, hidden_size)
        self.down_proj_norm = LayerNorm(hidden_size, eps=1e-12)

        self.act_fn = gelu
        self.dropout = Dropout(dropout_prob)

    def forward(self, former_output, hidden_states, row_ids, col_ids, sparse_attention_mask, attention_mask):
        # print('layer wise', hidden_states.shape, sparse_attention_mask.shape)
        # print('col ids:', col_ids.shape)
        table_mask = torch.where(col_ids == 0, torch.tensor(0.0).to(col_ids.device), torch.tensor(1.0).to(col_ids.device))
        table_mask = torch.unsqueeze(table_mask, dim=-1)

        row_hidden = self.row_adapter_layer(former_output, hidden_states, row_ids, attention_mask) * table_mask
        column_hidden = self.col_adapter_layer(former_output, hidden_states, col_ids, attention_mask) * table_mask

        layer_input = torch.cat([former_output, hidden_states, row_hidden, column_hidden], dim=-1)
        layer_input = self.dropout(self.act_fn(self.down_proj_layer(layer_input)))
        layer_input = former_output + layer_input
        layer_input = self.down_proj_norm(layer_input)

        sparse_attention_mask = torch.unsqueeze(sparse_attention_mask, dim=1)
        adapter_output = self.seq_adapter_layer(layer_input, attention_mask=sparse_attention_mask)
        # print(adapter_output)
        return adapter_output


class TAdapter(nn.Module):
    def __init__(self, layer_indices, hidden_size=1024, ff_dim=3072, dropout_prob=0.2):
        super(TAdapter, self).__init__()
        num_layer = len(layer_indices)

        self.layer_indices = layer_indices
        self.adapter_layers = nn.ModuleList()
        for k in range(num_layer):
            adapter_layer = TableAdapterLayer(hidden_size=hidden_size, ff_dim=ff_dim, dropout_prob=dropout_prob)
            self.adapter_layers.append(adapter_layer)

        self.row_proj_layer = ProjectionLayer(hidden_size=hidden_size)
        self.col_proj_layer = ProjectionLayer(hidden_size=hidden_size)
        self.col_wise_proj_layer = ProjectionLayer(hidden_size=hidden_size)

    def forward(self, embedding_output, all_layer_outputs, token_type_ids, sparse_attention_mask, attention_mask,
                return_row_col=False):
        row_ids = token_type_ids[:, :, 2]
        col_ids = token_type_ids[:, :, 1]
        former_output = embedding_output

        layer_outputs = [all_layer_outputs[i] for i in self.layer_indices]

        for lx, hidden_states in enumerate(layer_outputs):
            # print(lx)
            adapter_output = self.adapter_layers[lx](former_output=former_output,
                                                     hidden_states=hidden_states,
                                                     row_ids=row_ids,
                                                     col_ids=col_ids,
                                                     sparse_attention_mask=sparse_attention_mask,
                                                     attention_mask=attention_mask)
            # input()
            former_output = adapter_output

        if return_row_col is True:
            col_hidden_states = self.col_proj_layer(former_output, col_ids)
            row_hidden_states = self.row_proj_layer(former_output, row_ids)
            col_wise_hidden_states = self.col_wise_proj_layer(former_output, col_ids, with_cell_shape=True)

            return former_output, col_hidden_states, row_hidden_states, col_wise_hidden_states

        return former_output


class TableSpecificBERT(nn.Module):
    def __init__(self, hidden_size=1024, model_name='roberta'):
        super(TableSpecificBERT, self).__init__()
        if hidden_size == 768:
            if model_name == 'roberta':
                self.bert = AutoModel.from_pretrained('FacebookAI/xlm-roberta-large')
            else:
                self.bert = AutoModel.from_pretrained("FacebookAI/xlm-roberta-large")
        else:
            if model_name == 'roberta':
                self.bert = AutoModel.from_pretrained('FacebookAI/xlm-roberta-large')
            else:
                self.bert = AutoModel.from_pretrained("FacebookAI/xlm-roberta-large")
        if hidden_size == 1024:
            self.adapter_structure = TAdapter(layer_indices=[7, 15, -1], hidden_size=hidden_size)
        else:
            self.adapter_structure = TAdapter(layer_indices=[4, 8, -1], hidden_size=hidden_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = Dropout(0.1)

        # additional embeddings initialization
        max_vocab_sizes = [300, 300, 300, 300, 300]

        self.seg_embedding = nn.Embedding(max_vocab_sizes[0], hidden_size)
        self.col_embedding = nn.Embedding(max_vocab_sizes[1], hidden_size)
        self.row_embedding = nn.Embedding(max_vocab_sizes[2], hidden_size)

    def forward(self,
                input_ids,
                token_type_ids=None, attention_mask=None, table_ids=None, return_row_col=False):

        if table_ids is None:
            table_ids = torch.ones_like(input_ids)

        segment_ids = token_type_ids[:, :, 0]
        column_ids = token_type_ids[:, :, 1]
        row_ids = token_type_ids[:, :, 2]
        sparse_attention_mask = attention_utils.compute_sparse_attention(input_ids,
                                                                         segment_ids,
                                                                         row_ids,
                                                                         column_ids,
                                                                         table_ids)

        # make token type embedding with additional embeddings for adapter layers
        token_type_embedding_list = [
            self.seg_embedding,
            self.col_embedding,
            self.row_embedding,
        ]

        token_type_embeddings = None
        for i in range(len(token_type_embedding_list)):
            if i == 0:
                # print(input_ids.device, token_type_ids.device)
                token_type_embeddings = token_type_embedding_list[i](token_type_ids[:, :, i])
            else:
                token_type_embeddings += token_type_embedding_list[i](token_type_ids[:, :, i])

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        all_hidden_states = outputs.hidden_states
        hidden_states = outputs.last_hidden_state
        pooler_output = hidden_states[:, 0, :]

        if return_row_col is True:
            adapter_output, col_hidden_states, row_hidden_states, col_wise_hidden_states = self.adapter_structure(
                embedding_output=token_type_embeddings,
                all_layer_outputs=all_hidden_states,
                token_type_ids=token_type_ids,
                sparse_attention_mask=sparse_attention_mask,
                attention_mask=attention_mask,
                return_row_col=return_row_col
            )

            return hidden_states, adapter_output, pooler_output, col_hidden_states, row_hidden_states, col_wise_hidden_states
        else:
            adapter_output = self.adapter_structure(
                embedding_output=token_type_embeddings,
                all_layer_outputs=all_hidden_states,
                token_type_ids=token_type_ids,
                sparse_attention_mask=sparse_attention_mask,
                attention_mask=attention_mask
            )

            return hidden_states, adapter_output, pooler_output


class TableSpecificBERTOnlyProjection(nn.Module):
    def __init__(self, hidden_size=1024):
        super(TableSpecificBERTOnlyProjection, self).__init__()
        self.bert = AutoModel.from_pretrained('FacebookAI/roberta-large')
        self.col_projection_layer = ProjectionLayer(hidden_size=hidden_size)
        self.row_projection_layer = ProjectionLayer(hidden_size=hidden_size)
        self.projection_linear = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self,
                input_ids, position_ids=None,
                token_type_ids=None, attention_mask=None):

        column_ids = token_type_ids[:, :, 1]
        row_ids = token_type_ids[:, :, 2]

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)

        hidden_states = outputs.last_hidden_state
        pooler_output = hidden_states[:, 0, :]

        col_outputs = self.col_projection_layer(hidden_states, column_ids)
        row_outputs = self.col_projection_layer(hidden_states, row_ids)

        combined_hidden_states = torch.cat([hidden_states, col_outputs, row_outputs], dim=-1)
        combined_hidden_states = gelu(self.projection_linear(combined_hidden_states))

        return hidden_states, combined_hidden_states, pooler_output


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, :, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TableSpecificBERTTAdapter(nn.Module):
    def __init__(self):
        super(TableSpecificBERTTAdapter, self).__init__()
        self.bert = AutoModel.from_pretrained('FacebookAI/roberta-large')

        self.adapter_structure = TAdapter(layer_indices=[7, 15, -1])
        self.classifier = RobertaClassificationHead(self.bert.config)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = Dropout(0.1)

        # additional embeddings initialization
        max_vocab_sizes = [300, 300, 300, 300, 300]
        hidden_size = 1024

        self.seg_embedding = nn.Embedding(max_vocab_sizes[0], hidden_size)
        self.col_embedding = nn.Embedding(max_vocab_sizes[1], hidden_size)
        self.row_embedding = nn.Embedding(max_vocab_sizes[2], hidden_size)

        self.col_projection_layer = ProjectionLayer(hidden_size=hidden_size)
        self.row_projection_layer = ProjectionLayer(hidden_size=hidden_size)
        self.projection_linear = nn.Linear(hidden_size * 4, hidden_size)

        self.num_labels = 2

    def forward(self,
                input_ids,
                token_type_ids=None, attention_mask=None, position_ids=None,
                table_ids=None, labels=None):

        if table_ids is None:
            table_ids = torch.ones_like(input_ids)

        segment_ids = token_type_ids[:, :, 0]
        column_ids = token_type_ids[:, :, 1]
        row_ids = token_type_ids[:, :, 2]
        sparse_attention_mask = attention_utils.compute_sparse_attention(input_ids,
                                                                         segment_ids,
                                                                         row_ids,
                                                                         column_ids,
                                                                         table_ids)

        # make token type embedding with additional embeddings for adapter layers
        token_type_embedding_list = [
            self.seg_embedding,
            self.col_embedding,
            self.row_embedding,
        ]

        token_type_embeddings = None
        for i in range(len(token_type_embedding_list)):
            if i == 0:
                # print(input_ids.device, token_type_ids.device)
                token_type_embeddings = token_type_embedding_list[i](token_type_ids[:, :, i])
            else:
                token_type_embeddings += token_type_embedding_list[i](token_type_ids[:, :, i])
        # print(torch.max(segment_ids), torch.min(segment_ids))
        # print(attention_mask[0])
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        all_hidden_states = outputs.hidden_states
        hidden_states = outputs.last_hidden_state
        pooler_output = hidden_states[:, 0, :]

        # print(all_hidden_states[-1])
        # hidden_states = outputs.hidden_states
        # print(all_hidden_states[0].shape)
        # input()
        sparse_attention_mask = sparse_attention_mask.to(input_ids.device)
        adapter_output = self.adapter_structure(
            embedding_output=token_type_embeddings,
            all_layer_outputs=all_hidden_states,
            token_type_ids=token_type_ids,
            sparse_attention_mask=sparse_attention_mask,
            attention_mask=attention_mask
        )

        col_outputs = self.col_projection_layer(hidden_states, column_ids)
        row_outputs = self.col_projection_layer(hidden_states, row_ids)

        combined_hidden_states = torch.cat([hidden_states, adapter_output, col_outputs, row_outputs], dim=-1)
        combined_hidden_states = gelu(self.projection_linear(combined_hidden_states))

        return hidden_states, combined_hidden_states, pooler_output


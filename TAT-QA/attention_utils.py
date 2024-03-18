import numpy as np
import torch


def matches_token_type_id(tensor):
    return torch.eq(
        torch.unsqueeze(tensor, dim=1), torch.unsqueeze(tensor, dim=2)
    )


def matches_token_type_id_(tensor):
    return torch.eq(
        torch.unsqueeze(tensor, dim=0), torch.unsqueeze(tensor, dim=1)
    )


def get_attention_mask(input_ids):
    attention_mask = torch.where(input_ids > 0, torch.ones_like(input_ids), torch.zeros_like(input_ids))
    return attention_mask


def compute_sparse_attention(input_ids, segment_ids, row_ids, col_ids, table_ids):
    attention_mask = torch.where(input_ids > 0, torch.ones_like(input_ids), torch.zeros_like(input_ids))
    mask_one = torch.eq(attention_mask, 1)
    segment_zero = torch.eq(segment_ids, 0)

    sparse_attention_mask = matches_token_type_id(table_ids) & torch.unsqueeze(mask_one, dim=1) & (
        matches_token_type_id(col_ids) | matches_token_type_id(row_ids) |
        torch.unsqueeze(segment_zero, dim=2) |
        torch.unsqueeze(segment_zero, dim=1)
    )
    sparse_attention_mask = sparse_attention_mask.type(torch.FloatTensor)
    return sparse_attention_mask


def compute_column_row_attention(input_ids, table_ids, max_value=512):
    attention_mask = torch.where(input_ids < max_value, torch.ones_like(input_ids), torch.zeros_like(input_ids))
    mask_one = torch.eq(attention_mask, 1)

    sparse_attention_mask = matches_token_type_id_(table_ids) & torch.unsqueeze(mask_one, dim=1)
    sparse_attention_mask = sparse_attention_mask.type(torch.FloatTensor)
    return sparse_attention_mask
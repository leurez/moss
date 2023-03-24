#!/usr/bin/env python
#-*- coding:utf-8 -*-


import math
import torch

import torch.nn.functional as F
import moss.model.constant as const


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(
        const.OPENAI_GELU_TANH_P1 * x *(
            1.0 + const.OPENAI_GELU_TANH_P2 * x * x)))


def gelu(x):
    return gelu_impl(x)


def rotate_half(x):
    pivot = x.shape[-1] // 2
    front, tail = x[..., :pivot], x[..., pivot:]
    return torch.cat((-tail, front), dim=front.ndim -1)


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


def split_tensor_along_last_dim(
    tensor, num_partitions,contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, const.MASKED_FILL_VALUE)
    return attention_scores


def attention_fn(
    self, query_layer, key_layer, value_layer,
    attention_mask, hidden_size_per_partition, layer_id,
    layer_past=None, scaling_attention_score=True, use_cache=False):
    # main body of the func
    if layer_past is not None:
        past_key, past_value = layer_past
        key_layer = torch.cat((past_key, key_layer), dim=0)
        value_layer = torch.cat((past_value, value_layer), dim=0)
    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    seq_len, b, nh, hidden_size = key_layer.shape
    present = (key_layer, value_layer) if use_cache else None

    query_key_layer_scaling_coeff = layer_id + 1.0
    if scaling_attention_score:
        query_layer = query_layer / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)
    
    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

    matmul_result = torch.empty(
        output_size[0] * output_size[1],
        output_size[2],
        output_size[3],
        dtype=query_layer.dtype,
        device=query_layer.device,
    )

    matmul_result = torch.baddbmm(
        matmul_result,
        query_layer.transpose(0, 1),  # [b * np, sq, hn]
        key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        beta=0.0,
        alpha=1.0,
    )

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    if self.scale_mask_softmax:
        self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask.contiguous())
    else:
        if not (attention_mask == 0).all():
            # if auto-regressive, skip
            attention_scores.masked_fill_(
                attention_mask, const.MASKED_FILL_VALUE)
        dtype = attention_scores.type()
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = attention_probs.type(dtype)
    
    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + (hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, present, attention_probs)

    return outputs

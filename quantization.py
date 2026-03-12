import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_model_size_bytes(model: nn.Module) -> int:
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buf in model.buffers():
        total += buf.numel() * buf.element_size()
    return total


def _symmetric_quantize_per_row(weight: torch.Tensor, num_bits: int):
    if num_bits < 2 or num_bits > 8:
        raise ValueError(f"Only 2-8 bit symmetric quantization is supported, got {num_bits}")
    max_q = (1 << (num_bits - 1)) - 1
    scale = weight.detach().abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(scale / max(max_q, 1), min=1e-8)
    q_weight = torch.clamp(torch.round(weight.detach() / scale), -max_q, max_q).to(torch.int8)
    return q_weight, scale


class QuantizedLinear(nn.Module):
    def __init__(self, source: nn.Linear, num_bits: int):
        super().__init__()
        q_weight, scale = _symmetric_quantize_per_row(source.weight, num_bits)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.num_bits = num_bits
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale.to(torch.float32))
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (self.q_weight.to(torch.float32) * self.scale).to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


class QuantizedEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding, num_bits: int):
        super().__init__()
        q_weight, scale = _symmetric_quantize_per_row(source.weight, num_bits)
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.num_bits = num_bits
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        q_rows = self.q_weight.index_select(0, flat)
        scales = self.scale.index_select(0, flat)
        emb = q_rows.to(torch.float32) * scales
        emb = emb.reshape(*x.shape, self.embedding_dim)
        return emb


def _replace_with_quantized_modules(module: nn.Module, num_bits: int):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear(child, num_bits))
            continue
        if isinstance(child, nn.Embedding):
            setattr(module, name, QuantizedEmbedding(child, num_bits))
            continue
        _replace_with_quantized_modules(child, num_bits)


def make_quantized_copy(model: nn.Module, num_bits: int) -> nn.Module:
    quantized = copy.deepcopy(model).eval()
    _replace_with_quantized_modules(quantized, num_bits)
    for param in quantized.parameters():
        param.requires_grad_(False)
    return quantized

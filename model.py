import numpy as np
import torch
import torch.nn as nn


def _make_head(d_model: int, vocab_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, vocab_size),
    )


def _make_ffn(d_model: int, expansion: int = 4) -> nn.Sequential:
    hidden = expansion * d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Linear(hidden, d_model),
    )


def _bump_offset(inf, k: int = 1):
    if hasattr(inf, "seqlen_offset"):
        inf.seqlen_offset += k
    elif hasattr(inf, "sequence_length_offset"):
        setattr(inf, "sequence_length_offset", getattr(inf, "sequence_length_offset") + k)
    else:
        setattr(inf, "seqlen_offset", getattr(inf, "seqlen_offset", 0) + k)


def _tag_mamba_layers_with_ids(model, is_cuda: bool):
    idx = 0
    if is_cuda:
        from mamba_ssm import Mamba

        target_cls = Mamba
    else:
        from mambapy.mamba import MambaBlock as MambaCPU

        target_cls = MambaCPU

    for module in model.modules():
        if isinstance(module, target_cls):
            setattr(module, "layer_idx", idx)
            idx += 1


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, use_cuda: bool):
        super().__init__()
        self.use_cuda = use_cuda
        self.ln1 = nn.LayerNorm(d_model)
        if use_cuda:
            from mamba_ssm import Mamba

            self.mamba = Mamba(d_model=d_model)
        else:
            from mambapy.mamba import MambaBlock as MambaCPU, MambaConfig

            config = MambaConfig(d_model=d_model, n_layers=0, use_cuda=False)
            self.mamba = MambaCPU(config)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = _make_ffn(d_model)

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        if self.use_cuda:
            y = self.mamba(y, inference_params=inference_params)
        else:
            y = self.mamba(y)
        y = self.ln2(y)
        y = self.ff(y)
        return x + y

    def init_cache(self, batch_size: int, device):
        if self.use_cuda:
            raise RuntimeError("CUDA Mamba uses InferenceParams, not per-block caches")
        d_inner = self.mamba.config.d_inner
        d_conv = self.mamba.config.d_conv
        inputs = torch.zeros(batch_size, d_inner, d_conv - 1, device=device)
        return (None, inputs)

    def step(self, x, cache):
        if self.use_cuda:
            raise RuntimeError("Call BytewiseMamba.step on CUDA instead of block.step")
        y = self.ln1(x)
        y, cache = self.mamba.step(y, cache)
        y = self.ln2(y)
        y = self.ff(y)
        return x + y, cache


class BytewiseMamba(nn.Module):
    def __init__(self, d_model=256, num_layers=4, vocab_size=256, device="cuda"):
        super().__init__()
        self.is_cuda_backbone = torch.cuda.is_available() and device == "cuda"
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MambaResidualBlock(d_model, use_cuda=self.is_cuda_backbone) for _ in range(num_layers)]
        )
        self.head = _make_head(d_model, vocab_size)
        if self.is_cuda_backbone:
            from mamba_ssm.utils.generation import InferenceParams

            self._inference_params_cls = InferenceParams
        else:
            self._inference_params_cls = None
        _tag_mamba_layers_with_ids(self, self.is_cuda_backbone)

    def forward(self, x, inference_params=None):
        h = self.embedding(x)
        for blk in self.blocks:
            h = blk(h, inference_params=inference_params)
        return self.head(h)

    @torch.inference_mode()
    def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
        if self.is_cuda_backbone:
            return self._inference_params_cls(max_batch_size=batch_size, max_seqlen=max_len)
        stream_device = device or self.embedding.weight.device
        return [blk.init_cache(batch_size, stream_device) for blk in self.blocks]

    @torch.inference_mode()
    def step(self, byte_t: torch.LongTensor, stream_state) -> torch.Tensor:
        if self.is_cuda_backbone:
            x = self.embedding(byte_t).unsqueeze(1)
            h = x
            for blk in self.blocks:
                h = blk(h, inference_params=stream_state)
            logits_next = self.head(h).squeeze(1)
            _bump_offset(stream_state, 1)
            return logits_next

        h = self.embedding(byte_t)
        for i, blk in enumerate(self.blocks):
            h, stream_state[i] = blk.step(h, stream_state[i])
        return self.head(h)


class MinGRUCell(nn.Module):
    """Single-gate recurrent cell used as a lightweight quantization-friendly baseline."""

    def __init__(self, d_model: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.state_proj = nn.Linear(d_model, 2 * d_model, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gates = self.in_proj(x_t) + self.state_proj(h_prev)
        update, candidate = gates.chunk(2, dim=-1)
        update = torch.sigmoid(update)
        candidate = torch.tanh(candidate)
        return torch.lerp(h_prev, candidate, update)

    def forward_sequence(self, x: torch.Tensor, h_prev: torch.Tensor | None = None):
        batch, seq_len, width = x.shape
        if h_prev is None:
            h_prev = x.new_zeros(batch, width)
        outputs = torch.empty_like(x)
        h_t = h_prev
        for t in range(seq_len):
            h_t = self(x[:, t, :], h_t)
            outputs[:, t, :] = h_t
        return outputs, h_t


class MinGRUResidualBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.cell = MinGRUCell(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = _make_ffn(d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        recurrent_out, state = self.cell.forward_sequence(self.ln1(x), state)
        y = self.ln2(recurrent_out)
        y = self.ff(y)
        return x + y, state

    def init_cache(self, batch_size: int, device, dtype=None):
        dtype = dtype or torch.float32
        return torch.zeros(batch_size, self.ln1.normalized_shape[0], device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: torch.Tensor):
        recurrent_out = self.cell(self.ln1(x), state)
        y = self.ln2(recurrent_out)
        y = self.ff(y)
        return x + y, recurrent_out


class BytewiseMinGRU(nn.Module):
    def __init__(self, d_model=256, num_layers=4, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MinGRUResidualBlock(d_model) for _ in range(num_layers)])
        self.head = _make_head(d_model, vocab_size)

    def forward(self, x, inference_params=None):
        del inference_params
        h = self.embedding(x)
        states = [None] * len(self.blocks)
        for i, blk in enumerate(self.blocks):
            h, states[i] = blk(h, states[i])
        return self.head(h)

    @torch.inference_mode()
    def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
        del max_len
        stream_device = device or self.embedding.weight.device
        stream_dtype = dtype or self.embedding.weight.dtype
        return [blk.init_cache(batch_size, stream_device, stream_dtype) for blk in self.blocks]

    @torch.inference_mode()
    def step(self, byte_t: torch.LongTensor, stream_state) -> torch.Tensor:
        h = self.embedding(byte_t)
        for i, blk in enumerate(self.blocks):
            h, stream_state[i] = blk.step(h, stream_state[i])
        return self.head(h)


def BoaConstrictor(
    d_model=256,
    num_layers=4,
    vocab_size=256,
    device="cuda",
    backbone="mamba",
):
    backbone_name = str(backbone).lower()
    if backbone_name == "mamba":
        return BytewiseMamba(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
            device=device,
        )
    if backbone_name in {"mingru", "min_gru", "gru"}:
        return BytewiseMinGRU(
            d_model=d_model,
            num_layers=num_layers,
            vocab_size=vocab_size,
        )
    raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of: mamba, mingru")


def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    block = seq_len * batch_size
    return (n_bytes // block) * block


def make_splits(data_bytes: bytes | np.ndarray, seq_len: int, batch_size: int, splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    train_bytes = buf[i0:i1].tobytes()
    val_bytes = buf[i1:i2].tobytes()
    test_bytes = buf[i2 : i2 + n_test].tobytes()

    return train_bytes, val_bytes, test_bytes


class ByteDataloader:
    """Simple dataloader that yields batches of bytes."""

    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cuda"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pos = 0
        self.device = device

    def __len__(self):
        return len(self.data_bytes) // (self.seq_len * self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.seq_len * self.batch_size > len(self.data_bytes):
            self.pos = 0
            raise StopIteration

        batch_indices = np.arange(self.pos, self.pos + self.seq_len * self.batch_size)
        batch_indices = batch_indices.reshape(self.batch_size, self.seq_len)
        self.pos += self.seq_len * self.batch_size

        batch = self.data_bytes[batch_indices]
        return torch.tensor(batch, dtype=torch.long).to(self.device)

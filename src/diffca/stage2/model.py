"""Stage 2 separator: splits the K=32 conditioning tokens Z into a salient code and a common code.

Encoder: a CLS token is prepended to Z, a pre-norm Transformer encoder runs on the 33 tokens. The CLS output
h (the salient code, "cls" below) and the 32 other outputs (the common tokens) are read out.
Decoder (`SumDecoder`): Z_hat = common tokens + reshape(W h), with W a low-rank linear map R^d -> R^(32 x d).

Parameter and buffer names are those of the original code, so existing checkpoints load unchanged.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn

NUM_TOKENS = 32


def _ema_update(online: nn.Module, ema: nn.Module, decay: float):
    for online_param, ema_param in zip(online.parameters(), ema.parameters()):
        ema_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)


def _frozen_copy(module: nn.Module) -> nn.Module:
    copy = deepcopy(module)
    for param in copy.parameters():
        param.detach_()
    return copy


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to the CLS token and the 32 tokens."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = NUM_TOKENS + 1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class DisentanglerEncoder(nn.Module):
    """[CLS] + Z -> (salient code h = CLS output, common tokens).

    Keeps two frozen copies of the Transformer: an EMA copy used for evaluation (`use_ema`), and a faster EMA
    used to re-encode the swapped reconstructions in the cycle-consistency pass (`use_cycle_ema`).
    """

    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float,
                 ema_decay: float = 0.99, cycle_ema_decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True,
                                                   norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.ema_decay = ema_decay
        self.cycle_ema_decay = cycle_ema_decay
        self.ema_encoder = _frozen_copy(self.transformer_encoder)
        self.cycle_ema_encoder = _frozen_copy(self.transformer_encoder)

    @torch.no_grad()
    def update_ema(self):
        _ema_update(self.transformer_encoder, self.ema_encoder, self.ema_decay)

    @torch.no_grad()
    def update_cycle_ema(self):
        _ema_update(self.transformer_encoder, self.cycle_ema_encoder, self.cycle_ema_decay)

    def forward(self, src: torch.Tensor, use_ema: bool = False, use_cycle_ema: bool = False):
        cls_tokens = self.cls_token.expand(src.size(0), -1, -1)
        encoder_input = self.pos_encoder(torch.cat([cls_tokens, src], dim=1))
        if use_ema:
            encoder = self.ema_encoder
        elif use_cycle_ema:
            encoder = self.cycle_ema_encoder
        else:
            encoder = self.transformer_encoder
        encoded = encoder(encoder_input)
        return encoded[:, 0, :], encoded[:, 1:, :]


class SumDecoder(nn.Module):
    """Z_hat = common tokens + reshape(proj_up(proj_down(h))): the salient code enters additively."""

    def __init__(self, d_model: int = 128, ema_decay: float = 0.99, low_rank: int = 64):
        super().__init__()
        self.proj_down = nn.Linear(d_model, low_rank, bias=False)
        self.proj_up = nn.Linear(low_rank, d_model * NUM_TOKENS, bias=False)
        self.ema_decay = ema_decay
        self.ema_proj_down = _frozen_copy(self.proj_down)
        self.ema_proj_up = _frozen_copy(self.proj_up)

    @torch.no_grad()
    def update_ema(self):
        _ema_update(self.proj_down, self.ema_proj_down, self.ema_decay)
        _ema_update(self.proj_up, self.ema_proj_up, self.ema_decay)

    def forward(self, content_tokens: torch.Tensor, cls_token: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        """`cls_token`: (B, d) or (B, 1, d)."""
        proj_down = self.ema_proj_down if use_ema else self.proj_down
        proj_up = self.ema_proj_up if use_ema else self.proj_up
        projected_cls = proj_up(proj_down(cls_token))
        projected_cls = projected_cls.view(projected_cls.size(0), NUM_TOKENS, -1)
        return projected_cls + content_tokens


class TokenDisentangler(nn.Module):
    """Encoder + additive decoder.

    `cls_classifier` is a linear head on h. No loss trains it: it stays at its initialization and only feeds a
    validation printout. It is kept so that existing checkpoints load.
    Only the additive decoder of the paper is kept, so `num_decoder_layers` must be 0.
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 0,
                 dim_feedforward: int = 512, dropout: float = 0.1, ema_decay: float = 0.99,
                 cycle_ema_decay: float = 0.99, low_rank: int = 64):
        super().__init__()
        if num_decoder_layers != 0:
            raise ValueError("Only the additive SumDecoder (num_decoder_layers=0) is supported.")
        self.encoder = DisentanglerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                                           ema_decay=ema_decay, cycle_ema_decay=cycle_ema_decay)
        self.decoder = SumDecoder(d_model, ema_decay=ema_decay, low_rank=low_rank)
        self.cls_classifier = nn.Linear(d_model, 1)

    @torch.no_grad()
    def update_ema(self):
        self.encoder.update_ema()
        self.decoder.update_ema()

    @torch.no_grad()
    def update_cycle_ema(self):
        self.encoder.update_cycle_ema()

    def forward(self, src: torch.Tensor, labels: torch.Tensor | None = None, use_ema: bool = False):
        """Returns (reconstructed tokens, salient code h, common tokens, untrained head logits)."""
        cls, content_tokens = self.encoder(src, use_ema=use_ema)
        reconstructed_tokens = self.decoder(content_tokens, cls, use_ema=use_ema)
        predicted_y = self.cls_classifier(cls)
        return reconstructed_tokens, cls, content_tokens, predicted_y


class CycleConsistencyWrapper(nn.Module):
    """Runs the separator and, with `cycle=True`, the cycle-consistency pass.

    Cycle pass: the batch is split into pairs (2i, 2i+1); each sample is decoded with the salient code of its
    partner, and the result is re-encoded with the cycle-EMA encoder. `encoder` and `decoder` are aliases of
    `model.encoder` / `model.decoder` (they also appear in the state dict).
    """

    def __init__(self, disentangler_model: nn.Module):
        super().__init__()
        self.model = disentangler_model
        self.encoder = disentangler_model.encoder
        self.decoder = disentangler_model.decoder

    @torch.no_grad()
    def update_ema(self):
        self.model.update_ema()

    @torch.no_grad()
    def update_cycle_ema(self):
        self.model.update_cycle_ema()

    def forward(self, src: torch.Tensor, labels: torch.Tensor = None, cycle=False, use_ema: bool = False) -> dict:
        reconstructed_tokens, cls, content_tokens, predicted_y = self.model(src, labels, use_ema=use_ema)

        cycle_outputs = None
        if cycle:
            # An odd last sample is left out of the pairs.
            cycle_batch_size = src.size(0) - (src.size(0) % 2)
            even = torch.arange(0, cycle_batch_size, 2, device=src.device)
            odd = torch.arange(1, cycle_batch_size, 2, device=src.device)
            cls_even, cls_odd = cls[:cycle_batch_size][even], cls[:cycle_batch_size][odd]
            content_even, content_odd = content_tokens[:cycle_batch_size][even], content_tokens[:cycle_batch_size][odd]

            swapped_even = self.decoder(content_even, cls_odd)  # common of 2i, salient of 2i+1
            swapped_odd = self.decoder(content_odd, cls_even)
            cycled_cls_even, cycled_content_even = self.encoder(swapped_even, use_cycle_ema=True)
            cycled_cls_odd, cycled_content_odd = self.encoder(swapped_odd, use_cycle_ema=True)

            cycle_outputs = {
                "original_a": {"cls": cls_even, "content": content_even},
                "original_b": {"cls": cls_odd, "content": content_odd},
                "cycled_a": {"cls": cycled_cls_even, "content": cycled_content_even},
                "cycled_b": {"cls": cycled_cls_odd, "content": cycled_content_odd},
            }

        standard_outputs = {
            "reconstructed_tokens": reconstructed_tokens,
            "cls": cls,
            "predicted_y": predicted_y,
            "content_tokens": content_tokens,
            "original_src": src,
            "labels": labels,
        }
        return {"standard_outputs": standard_outputs, "cycle_outputs": cycle_outputs}


class AdversaryMLP(nn.Module):
    """Predicts the label Y from the flattened common tokens.

    Its buffers hold the state of the self-tuning gradient reversal (see `losses.compute_adversarial_loss`):
    `ema_acc`, `lambda_val`, `adaptive_mode`. The other buffers belong to removed options and are never
    updated; they are kept so that existing checkpoints load.
    """

    def __init__(self, num_tokens: int = 32, token_dim: int = 128, hidden_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.register_buffer("ema_acc", torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer("lambda_val", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("adaptive_mode", torch.tensor(False, dtype=torch.bool))
        # Unused (removed push loss and PI controller).
        self.register_buffer("ema_push_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("lambda_push_val", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("adaptive_push_mode", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("integral_error", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ki", torch.tensor(0.001, dtype=torch.float32))
        self.register_buffer("kp", torch.tensor(0.01, dtype=torch.float32))

        self.classifier = nn.Sequential(
            nn.Linear(num_tokens * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(context_tokens.view(context_tokens.shape[0], -1))

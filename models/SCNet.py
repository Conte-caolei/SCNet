"""
SCNet: Explicit Same-Scale Cross-Channel Modeling for
Multi-Scale Multivariate Time Series Forecasting.

Modules (aligned with the paper):
    - MAPE : Multi-Scale Adaptive Patch Embedding        (Sec. III-B, Eq. 1-2)
    - CSAL : Cross-Channel Same-Scale Association Learner (Sec. III-C, Eq. 3-6)
    - TDL  : Temporal Dependency Learner                  (Sec. III-D, Eq. 7-9)
    - SGF  : Scale Gate Fusion                            (Sec. III-E, Eq. 10-13)

Notation: B = batch, C = channels, S = number of scales,
          D = d_model, L = look-back length, T = prediction horizon.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import EmbLayer
from layers.RevIN import RevIN
from utils.CKA import CudaCKA


class MAPE(nn.Module):
    """Multi-Scale Adaptive Patch Embedding.

    Performs multi-scale decomposition and scale-adaptive patch embedding:
        z^(s) = PatchEmbed(x^(s), p_s)                          (Eq. 2)
    Smaller patches are used at fine-grained scales and larger patches
    at coarse-grained scales.
    """

    def __init__(self, seq_len, d_model, patch_sizes):
        super().__init__()
        self.patch_embeds = nn.ModuleList([
            EmbLayer(p, p // 2, seq_len, d_model)
            for p in patch_sizes
        ])

    def forward(self, x):
        # x: [B, C, L]
        z = [embed(x) for embed in self.patch_embeds]  # S x [B, C, D]
        return torch.stack(z, dim=2)                   # [B, C, S, D]


class CSAL(nn.Module):
    """Cross-Channel Same-Scale Association Learner.

    Explicitly models cross-channel dependencies within each temporal scale
    via a constrained and normalized linear mapping:
        W_hat  = row-wise L1-normalized Softplus(W)             (Eq. 4)
        Z_tilde = W_hat @ Z                                     (Eq. 5)
        Z'      = LayerNorm(Z + Linear(Z_tilde))                (Eq. 6)
    """

    def __init__(self, d_model, num_channels, dropout=0.1):
        super().__init__()
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # W in Eq. (4): learnable cross-channel association matrix
        init_weight = torch.eye(num_channels) + torch.randn(num_channels, num_channels)
        self.channel_weight = nn.Parameter(init_weight[None, :, :])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        # z: [B, C, S, D]
        values = self.v_proj(z)
        w_hat = F.softplus(self.channel_weight)        # non-negativity
        w_hat = F.normalize(w_hat, p=1, dim=-1)        # row-wise L1 normalization
        w_hat = self.dropout(w_hat).squeeze(0)         # [C, C]
        z_tilde = torch.einsum('ij,bjsd->bisd', w_hat, values)  # Eq. (5)
        return self.norm(z + self.dropout(self.out_proj(z_tilde)))  # Eq. (6)


class TDL(nn.Module):
    """Temporal Dependency Learner.

    Captures channel-wise temporal dynamics with a lightweight
    feed-forward transformation:
        H     = GELU(Linear_1(Z'))                              (Eq. 7)
        H_hat = Linear_2(H)                                     (Eq. 8)
        Y     = LayerNorm(Z' + H_hat)                           (Eq. 9)
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # 1x1 convolutions are equivalent to Linear_1 / Linear_2 in Eq. (7)-(8)
        self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.linear2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        # z: [B, C, S, D]
        B, C, S, D = z.shape
        h = z.reshape(B * C * S, D).unsqueeze(-1)      # [B*C*S, D, 1]
        h = self.dropout(self.activation(self.linear1(h)))   # Eq. (7)
        h_hat = self.dropout(self.linear2(h))                 # Eq. (8)
        h_hat = h_hat.squeeze(-1).reshape(B, C, S, D)
        return self.norm(z + h_hat)                           # Eq. (9)


class SCNetEncoderLayer(nn.Module):
    """One SCNet block: CSAL followed by TDL, applied per temporal scale."""

    def __init__(self, d_model, d_ff, num_channels, dropout=0.1, activation="gelu"):
        super().__init__()
        self.csal = CSAL(d_model, num_channels, dropout)
        self.tdl = TDL(d_model, d_ff, dropout, activation)

    def forward(self, z):
        return self.tdl(self.csal(z))


class SCNetEncoder(nn.Module):
    """Stack of SCNet encoder layers with a final LayerNorm."""

    def __init__(self, layers, norm_layer=None, CKA_flag=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.CKA_flag = CKA_flag
        if self.CKA_flag:
            print('CKA is enabled...')

    def forward(self, z):
        # z: [B, C, S, D]
        z0 = None
        layer_len = len(self.layers)
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if not self.training and self.CKA_flag and layer_len > 1:
                if i == 0:
                    z0 = z
                if i == layer_len - 1 and random.uniform(0, 1) < 1e-1:
                    cka = CudaCKA(device=z.device)
                    cka_value = cka.linear_CKA(z0.flatten(0, 1)[:1000], z.flatten(0, 1)[:1000])
                    print(f'CKA: \t{cka_value:.3f}')

        if self.norm is not None:
            z = self.norm(z)
        return z


class SGF(nn.Module):
    """Scale Gate Fusion.

    Adaptively integrates multi-scale features via a gating mechanism:
        g_c^(s) = sigmoid(W_g Y_c^(s))                          (Eq. 11)
        F_c     = sum_s g_c^(s) * Y_c^(s)                       (Eq. 12)

    NOTE: 若需复现消融实验 "w/o SGF"（均匀平均融合），
    将 forward 替换为: return y.mean(dim=2)
    """

    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1)  # W_g in Eq. (11)

    def forward(self, y):
        # y: [B, C, S, D]
        g = torch.sigmoid(self.gate_proj(y))    # [B, C, S, 1]
        return (g * y).sum(dim=2)               # [B, C, D]


class Model(nn.Module):
    """SCNet: MAPE -> (CSAL -> TDL) x e_layers -> SGF -> linear predictor."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # number of channels C

        # Multi-Scale Adaptive Patch Embedding
        self.mape = MAPE(configs.seq_len, configs.d_model, configs.patch_sizes)

        # Stacked (CSAL -> TDL) blocks
        self.encoder = SCNetEncoder(
            [
                SCNetEncoderLayer(
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    num_channels=self.enc_in,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            CKA_flag=configs.CKA_flag,
        )

        # Scale Gate Fusion
        self.sgf = SGF(d_model=configs.d_model)

        # Linear predictor (Eq. 13)
        self.predictor = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.pred_len),
        )

        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

    def forecast(self, x_enc):
        # x_enc: [B, L, C]
        x_enc = self.revin_layer(x_enc, mode='norm')
        x = x_enc.permute(0, 2, 1)      # [B, C, L]

        z = self.mape(x)                # [B, C, S, D]
        y = self.encoder(z)             # [B, C, S, D]
        f = self.sgf(y)                 # [B, C, D]

        dec_out = self.predictor(f)     # [B, C, T]
        dec_out = dec_out.permute(0, 2, 1)  # [B, T, C]
        dec_out = self.dropout(dec_out)
        dec_out = self.revin_layer(dec_out, mode='denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, T, C]

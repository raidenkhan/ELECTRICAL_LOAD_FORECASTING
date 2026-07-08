"""Deep learning architectures for time series forecasting.

All models accept (batch, seq_len, n_features) and output (batch, forecast_horizon).
Designed for GPU throughput with large batch sizes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Positional Encoding (shared by Transformer / any attention model) ──

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


# ── Feature Projector (embed raw features into model dimension) ──

class FeatureEmbedding(nn.Module):
    """Projects n_features -> d_model."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)

    def forward(self, x):
        return self.proj(x)


# ── 1. LSTM ──

class LSTM(nn.Module):
    def __init__(self, n_features: int, forecast_horizon: int,
                 hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, forecast_horizon),
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)       # lstm_out: (B, S, H)
        last = lstm_out[:, -1, :]                  # (B, H)
        return self.head(last)                     # (B, 24)


# ── 2. GRU ──

class GRU(nn.Module):
    def __init__(self, n_features: int, forecast_horizon: int,
                 hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, forecast_horizon),
        )

    def forward(self, x):
        _, h_n = self.gru(x)                       # h_n: (L, B, H)
        last = h_n[-1]                             # (B, H)
        return self.head(last)


# ── 3. Transformer Encoder ──

class Transformer(nn.Module):
    def __init__(self, n_features: int, forecast_horizon: int,
                 d_model: int = 128, nhead: int = 4,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.embed = FeatureEmbedding(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, forecast_horizon),
        )

    def forward(self, x):
        x = self.embed(x)                          # (B, S, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                    # (B, S, d_model)
        pooled = x.mean(dim=1)                     # (B, d_model) — global average pooling over sequence
        return self.head(pooled)


# ── 4. TCN (Temporal Convolutional Network) ──

class Chomp1d(nn.Module):
    """Remove padding elements at the end to keep causal convolution."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """Dilated causal convolution block with residual."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        out = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.chomp2(self.conv2(out)))
        return self.act2(out + self.residual(x))


class TCN(nn.Module):
    def __init__(self, n_features: int, forecast_horizon: int,
                 channels: list = None, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        if channels is None:
            channels = [n_features, 128, 64, 64, 32]
        self.forecast_horizon = forecast_horizon

        # Input projection: n_features -> channels[0]
        self.input_proj = nn.Conv1d(n_features, channels[0], kernel_size=1) if channels[0] != n_features else nn.Identity()

        # Build TCN blocks
        blocks = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:], 1):
            dilation = 2 ** i
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.final_channels = channels[-1]

        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.GELU(),
            nn.Linear(channels[-1], forecast_horizon),
        )

    def forward(self, x):
        # x: (B, S, F) -> permute to (B, F, S) for Conv1d
        x = x.permute(0, 2, 1).contiguous()        # (B, F, S)
        x = self.input_proj(x)
        x = self.tcn(x)                             # (B, C, S)
        # Global max pooling over sequence
        x = x.max(dim=-1).values                    # (B, C)
        return self.head(x)


# ── 5. DLinear (from "Are Transformers Effective for Time Series Forecasting?") ──

class DLinear(nn.Module):
    """Decomposition-Linear: moving avg decomposition + 2 linear branches."""

    def __init__(self, n_features: int, forecast_horizon: int,
                 input_window: int, kernel: int = 25):
        super().__init__()
        # DLinear operates on the raw demand series and calendar features separately
        self.forecast_horizon = forecast_horizon
        self.input_window = input_window
        self.kernel = kernel

        n_demand = 1      # demand_mw column
        n_calendar = n_features - 1  # everything else

        # Seasonal + trend linear layers for demand channel
        self.trend_linear = nn.Linear(input_window, forecast_horizon)
        self.seasonal_linear = nn.Linear(input_window, forecast_horizon)

        # Calendar features get their own linear mapping
        self.calendar_linear = nn.Linear(input_window * n_calendar, forecast_horizon)

    def moving_avg(self, x):
        """x: (B, S). Returns trend (moving avg) and seasonal (residual)."""
        pad_left = self.kernel - 1
        pad_right = 0
        x_padded = F.pad(x.unsqueeze(1), (pad_left, pad_right), mode='replicate').squeeze(1)
        # x_padded: (B, S + kernel - 1)
        kernel = torch.ones(self.kernel, device=x.device) / self.kernel
        trend = F.conv1d(
            x_padded.unsqueeze(1),
            kernel.view(1, 1, -1),
            padding=0
        ).squeeze(1)                                    # (B, S)
        seasonal = x - trend
        return trend, seasonal

    def forward(self, x):
        # x: (B, S, F) where F = n_features
        demand = x[:, :, 0]                              # (B, S) — demand_mw is first feature
        calendar = x[:, :, 1:]                           # (B, S, F-1)

        # Decompose demand
        trend, seasonal = self.moving_avg(demand)        # both (B, S)

        trend_out = self.trend_linear(trend)             # (B, 24)
        seasonal_out = self.seasonal_linear(seasonal)    # (B, 24)

        # Calendar: flatten and project
        B, S, C = calendar.shape
        calendar_flat = calendar.reshape(B, S * C)       # (B, S*(F-1))
        calendar_out = self.calendar_linear(calendar_flat)  # (B, 24)

        return trend_out + seasonal_out + calendar_out


# ── Model Registry ──

MODEL_REGISTRY = {
    'lstm': LSTM,
    'gru': GRU,
    'transformer': Transformer,
    'tcn': TCN,
    'dlinear': DLinear,
}


def create_model(name: str, n_features: int, forecast_horizon: int, input_window: int, cfg) -> nn.Module:
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_REGISTRY.keys())}")

    kwargs = dict(n_features=n_features, forecast_horizon=forecast_horizon)

    if name in ('lstm', 'gru'):
        kwargs.update(hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif name == 'transformer':
        kwargs.update(d_model=cfg.d_model, nhead=cfg.nhead,
                       num_encoder_layers=cfg.num_encoder_layers,
                       dim_feedforward=cfg.dim_feedforward, dropout=cfg.transformer_dropout)
    elif name == 'tcn':
        kwargs.update(channels=cfg.tcn_channels, kernel_size=cfg.tcn_kernel_size, dropout=cfg.dropout)
    elif name == 'dlinear':
        kwargs.update(input_window=input_window, kernel=cfg.dlinear_kernel)

    model = cls(**kwargs)
    return model

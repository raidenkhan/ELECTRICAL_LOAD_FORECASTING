import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for stabilizing non-stationary time series.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        self.mean = None
        self.std = None

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)
        return x

    def _denormalize(self, x):
        if self.mean is None or self.std is None:
            return x
        if self.affine:
            x = (x - self.affine_bias.view(1, 1, -1)) / self.affine_weight.view(1, 1, -1)
        x = x * self.std + self.mean
        return x

class FutureFeatureProjector(nn.Module):
    def __init__(self, future_dim: int, d_model: int):
        super(FutureFeatureProjector, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(future_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, future_feats):
        return self.proj(future_feats)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.kernel_size = kernel_size

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        x_perm = x_pad.permute(0, 2, 1) # [Batch, D_Model, Seq_Len+Pad]
        trend = self.moving_avg(x_perm).permute(0, 2, 1)
        
        if trend.shape[1] > x.shape[1]:
            trend = trend[:, :x.shape[1], :]
        
        seasonal = x - trend
        return seasonal, trend

class OptimizedAutoformer(nn.Module):
    """
    Renamed from EnhancedAutoformer to match production naming.
    """
    def __init__(self, input_dim, future_dim, seq_len, horizon, 
                 d_model=32, n_heads=2, e_layers=1, dropout=0.1, 
                 use_revin=True):
        super(OptimizedAutoformer, self).__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.use_revin = use_revin
        
        if use_revin:
            self.revin = RevIN(1)
        
        # Consistent kernel size with training
        self.decomp = SeriesDecomp(kernel_size=25)
        
        self.enc_embedding = nn.Linear(input_dim, d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=e_layers)
        
        self.future_proj = FutureFeatureProjector(future_dim, d_model)
        
        self.seasonal_head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, horizon)
        )
        
        self.trend_head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, horizon)
        )
        
        self.future_refinement = nn.Linear(d_model, 1)
        
    def forward(self, x, future_feats=None, target_for_revin=None):
        batch_size = x.shape[0]
        
        if self.use_revin and target_for_revin is not None:
            target_normalized = self.revin(target_for_revin, mode='norm')
            x = x.clone()
            x[:, :, 0] = target_normalized.squeeze(-1)
        
        seasonal, trend = self.decomp(x)
        
        s_enc = self.enc_embedding(seasonal)
        t_enc = self.enc_embedding(trend)
        
        s_out = self.encoder(s_enc)
        t_out = self.encoder(t_enc)
        
        s_flat = s_out.reshape(batch_size, -1)
        t_flat = t_out.reshape(batch_size, -1)
        
        pred_seasonal = self.seasonal_head(s_flat)
        pred_trend = self.trend_head(t_flat)
        
        if future_feats is not None:
            future_context = self.future_proj(future_feats)
            future_adjustment = self.future_refinement(future_context).squeeze(-1)
            pred_combined = pred_seasonal + pred_trend + future_adjustment
        else:
            pred_combined = pred_seasonal + pred_trend
        
        if self.use_revin and target_for_revin is not None:
            pred_combined_reshaped = pred_combined.unsqueeze(-1)
            pred_combined_reshaped = self.revin(pred_combined_reshaped, mode='denorm')
            pred_combined = pred_combined_reshaped.squeeze(-1)
        
        return pred_combined

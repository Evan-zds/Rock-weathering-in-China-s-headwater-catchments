import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class UnifiedResModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, r_ff, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.body = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * r_ff), nn.ReLU(), nn.BatchNorm1d(hidden_dim * r_ff), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim * r_ff, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout)
                                 )
    def forward(self, x):
        x_projected = self.proj(x)  # (B, D)
        return x_projected + self.body(x_projected)  # (B, D)


class KMLP(nn.Module):
    def __init__(self, feature_names=None, ndim=64, r_ff=4, dropout=0.1):
        super(KMLP, self).__init__()
        if feature_names:
            self.feature_names = feature_names

        params = {'hidden_dim': ndim, 'r_ff': r_ff, 'dropout': dropout}

        self.pCO2soil = UnifiedResModule(input_dim=2, **params)
        self.CO2air = UnifiedResModule(input_dim=1, **params)
        self.landcover = UnifiedResModule(input_dim=3, **params)
        self.AS = UnifiedResModule(input_dim=3 * ndim, **params)
        self.Slope_Esoil_Depthsoil = UnifiedResModule(input_dim=3, **params)
        self.Lithology = UnifiedResModule(input_dim=3, **params)
        self.MS = UnifiedResModule(input_dim=2 * ndim, **params)
        self.RM = UnifiedResModule(input_dim=2, **params)
        self.saturation_status = UnifiedResModule(input_dim=2, **params)
        self.Tsoil = UnifiedResModule(input_dim=1, **params)
        self.KC = UnifiedResModule(input_dim=2 * ndim, **params)
        self.HT = UnifiedResModule(input_dim=2, **params)
        self.SL = UnifiedResModule(input_dim=1 * ndim, **params)
        self.WL = UnifiedResModule(input_dim=4 * ndim, **params)
        self.gate_logic = nn.Sequential(
            nn.Linear(2 * ndim, ndim),
            nn.ReLU(),
            nn.Linear(ndim, ndim),
            nn.Sigmoid()  # 将输出限制在 (0, 1) 区间，作为权重
        )
        self.regression_head = nn.Sequential(
            nn.Linear(ndim, ndim // r_ff),
            nn.ReLU(),
            nn.BatchNorm1d(ndim // r_ff),
            nn.Dropout(dropout),
            nn.Linear(ndim // r_ff, ndim // (r_ff ** 2)),
            nn.ReLU(),
            nn.BatchNorm1d(ndim // (r_ff ** 2)),
            nn.Dropout(dropout),
            nn.Linear(ndim // (r_ff ** 2), 2)
        )

    def forward(self, x):
        feature_map = self.feature_names

        GPP_as_tensor = x[:, feature_map['GPP_as']].view(-1, 1)
        SOC_as_tensor = x[:, feature_map['SOC_as']].view(-1, 1)
        pCO2_out = self.pCO2soil(torch.cat([GPP_as_tensor, SOC_as_tensor], dim=-1))

        CO2air_as_out = self.CO2air(x[:, feature_map['CO2air_as']].view(-1, 1))

        fCrop_as_tensor = x[:, feature_map['fCrop_as']].view(-1, 1)
        fForest_as_tensor = x[:, feature_map['fForest_as']].view(-1, 1)
        fGrass_as_tensor = x[:, feature_map['fGrass_as']].view(-1, 1)
        land_out = self.landcover(torch.cat([fCrop_as_tensor, fForest_as_tensor, fGrass_as_tensor], dim=-1))

        AS_in = torch.cat([pCO2_out, CO2air_as_out, land_out], dim=-1)
        AS_out = self.AS(AS_in)

        Slope_ms_tensor = x[:, feature_map['Slope_ms']].view(-1, 1)
        Depsoil_ms_tensor = x[:, feature_map['Depsoil_ms']].view(-1, 1)
        Esoil_ms_tensor = x[:, feature_map['Esoil_ms']].view(-1, 1)
        Slope_Esoil_Depsoil_out = self.Slope_Esoil_Depthsoil(torch.cat([Slope_ms_tensor, Esoil_ms_tensor, Depsoil_ms_tensor], dim=-1))
        fCa_ms_tensor = x[:, feature_map['fCa_ms']].view(-1, 1)
        fSi_ms_tensor = x[:, feature_map['fSi_ms']].view(-1, 1)
        kref_ms_tensor = x[:, feature_map['kref_ms']].view(-1, 1)
        Litho_out = self.Lithology(torch.cat([fCa_ms_tensor, fSi_ms_tensor, kref_ms_tensor], dim=-1))
        MS_out = self.MS(torch.cat([Slope_Esoil_Depsoil_out, Litho_out], dim=-1))

        Tsoil_kc_tensor = x[:, feature_map['Tsoil_kc']].view(-1, 1)
        pHsoil_kc_tensor = x[:, feature_map['pHsoil_kc']].view(-1, 1)
        sat_out = self.saturation_status(torch.cat([Tsoil_kc_tensor, pHsoil_kc_tensor], dim=-1))

        Tsoil_out = self.Tsoil(Tsoil_kc_tensor)

        KC_in = torch.cat([Tsoil_out, sat_out], dim=-1)
        KC_out = self.KC(KC_in)

        fClay_rm_tensor = x[:, feature_map['fClay_rm']].view(-1, 1)
        SM_rm_tensor = x[:, feature_map['SM_rm']].view(-1, 1)
        RM_out = self.RM(torch.cat([fClay_rm_tensor, SM_rm_tensor], dim=-1))

        ROsur_ht_tensor = x[:, feature_map['ROsur_ht']].view(-1, 1)
        ROsub_ht_tensor = x[:, feature_map['ROsub_ht']].view(-1, 1)
        HT_out = self.HT(torch.cat([ROsur_ht_tensor, ROsub_ht_tensor], dim=-1))

        SL_in = MS_out.clone()
        SL_out = self.SL(SL_in)

        WL_in = torch.cat([AS_out, KC_out, RM_out, HT_out], dim=-1)
        WL_out = self.WL(WL_in)

        gate_input = torch.cat([SL_out, WL_out], dim=-1)
        gate_weight = self.gate_logic(gate_input)
        fusion_feat = gate_weight * SL_out + (1 - gate_weight) * WL_out

        final_pred = self.regression_head(fusion_feat)

        final_outputs = torch.clamp(final_pred, 0, None)
        return final_outputs


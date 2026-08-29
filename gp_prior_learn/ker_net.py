from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .kernel import IKer, MultiTaskRBFKer
from .utils import MLP, ParamPSDMat, ParamDiagMat, ParamPDMat


# MLP映射后取高斯核
class MlpRBFKer(IKer):
    """MLP + RBF kernrl （i.e. deep kernel）: K_{rbf}(net(x), net(x^\prime)) 先用MLP把x映射到新的空间，再在这个空间用RBF核"""
    def __init__(self, args: argparse.Namespace):
        super().__init__(nf=args.dy, nz=args.dz, jitter=args.jitter)

        self.phi = MLP(
            di=args.dx, do=args.dz, s=32,
            dropout=getattr(args, "dropout", 0.0))

        std = args.std if hasattr(args, 'std') else [1. for _ in range(args.dy)]
        ls = args.ls if hasattr(args, 'ls') else [1. for _ in range(args.dz)]
        self.ker = MultiTaskRBFKer(nf=args.dy, nz=args.dz, std=std, ls=ls, jitter=args.jitter)

    def forward(self, Z1: Tensor, Z2: Optional[Tensor]=None) -> Tensor:
        """
        Args:
            Z1: (*, N1, nz)
            Z2: (*, N2, nz)
        Returns:
            K (*, N1*nf, N2*nf)
        """

        Z1_ = self.phi(Z1)
        Z2_ = self.phi(Z2) if Z2 is not None else Z2
        return self.ker(Z1_, Z2_)

    @property
    def ls(self):
        return self.ker.ls

    @property
    def var(self):
        return self.ker.var


# 未定义init会自动调用父类的init
class CustomEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        src2, attn = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn_weights = attn  # 注意力保存下来

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# 提取注意力运算结果构造正定核
class TfKer(nn.Module):
    """Kernel paramterized by transformer"""
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

        nhead = 8
        d_model = args.dx * nhead * 16

        self.mlp = MLP(
            di=args.dx, do=d_model, s=32,
            dropout=getattr(args, "dropout", 0.0))
        self.encoder_layer = CustomEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)


    def forward(self, Z1: Tensor, Z2: Optional[Tensor]=None) -> Tensor:
        """
       Args:
           Z1: (*, N1, nz)
           Z2: (*, N2, nz)
       Returns:
           K (*, N1*nf, N2*nf)
       """
        # 学不了Z1不等于Z2的情况，只能Z1和Z2先合并然后再取其中对应分块

        B, N, D = Z1.shape  # batch, length, latent dim

        # run transformer
        Z1_ = self.mlp(Z1)
        _ = self.encoder(Z1_)
        A = self.encoder.layers[-1].last_attn_weights
        # 提取注意力运算结果矩阵 A = softmax(QK^T/sqrt(d_k))，取最后一层

        # case 1
        # A = A.view(B, -1, N, N).mean(dim=1)                             # (B, N, N)
        # L = torch.tril(A)
        # diag_idx = torch.arange(N, device=Z1.device)
        # L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx])
        # K = L @ L.transpose(-1, -2)                                     # (B, N, N)

        # case 2
        # A = A.view(B, -1, N, N).mean(dim=1)  # (B, N, N)
        # K = A @ A.transpose(-1, -2)

        # case 3
        A = A.view(B, -1, N, N).mean(dim=1)
        L = torch.tril(A)
        # 提取左下三角
        K = L @ L.transpose(-1, -2)

        return K

# MLP映射后的二次型
class BfnKer(IKer):
    """Basis function kernel"""
    def __init__(self, args: argparse.Namespace):
        super().__init__(df=args.dy, dz=args.dx, jitter=args.jitter)

        self.args = args
        self.phi = MLP(
            di=args.dx, do=args.dz, s=32, activation=nn.GELU,
            dropout=args.dropout)
        # self.P0_param = ParamDiagMat(x=torch.eye(args.dz), is_train=True)
        self.P0_param = ParamDiagMat(x=torch.eye(args.dz), is_train=True, eps=args.jitter)
        # 从单位矩阵进行优化
    def forward(self, Z1: Tensor, Z2: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            Z1: (*, N1, nz)
            Z2: (*, N2, nz)
        Returns:
            K (*, N1*nf, N2*nf)
        """
        if Z2 is None:
            Z2 = Z1

        phi_1 = self.phi(Z1)    # (*, N1, db)
        phi_2 = self.phi(Z2)    # (*, N2, db)
        P0 = self.P0.to(Z1.device)

        K = phi_1 @ P0 @ phi_2.transpose(-1, -2) # (*, N1, N2)

        return K

    @property
    def P0(self):
        return self.P0_param()

    @property
    def ls(self) -> Tensor:
        return torch.ones(self.dz)

    @property
    def var(self) -> Tensor:
        return torch.ones(self.df)

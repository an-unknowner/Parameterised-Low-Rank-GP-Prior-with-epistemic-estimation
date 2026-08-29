from typing import Tuple, Optional, List, Union, OrderedDict as OrderedDictType
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, dim_feature, eps=1e-5, n_forget=500):
        """
        Custom Batch Normalization with exponential moving average.

        Args:
            dim_feature (int): Feature dimension to normalize.
            eps (float): Small constant for numerical stability.
            n_forget (int): Number of steps to reduce influence by 1 order of magnitude (used for momentum).
        """
        super().__init__()
        self.d = dim_feature
        self.eps = eps
        self.forget = 0.1 ** (1 / n_forget)  # Decay factor for moving average

        # Running statistics (not trainable)
        self.register_buffer("x_mean", torch.zeros(self.d))
        self.register_buffer("x_std", torch.ones(self.d))

    def forward(self, x):
        """
        Apply batch normalization to input tensor.

        Args:
            x (Tensor): Input of shape (*, d)

        Returns:
            Tensor: Normalized input, same shape as input.
        """
        x_flat = x.reshape(-1, self.d)  # Flatten batch dimensions

        # Compute current batch statistics
        x_mean = x_flat.mean(dim=0).detach()                 # (d,)
        x_std = x_flat.std(dim=0, unbiased=False) .detach()  # (d,)

        # Update running statistics
        self.update_stats(x_mean, x_std)

        # Normalize input using running statistics
        xn = (x - self.x_mean) / (self.x_std ** 2 + self.eps).sqrt()

        return xn

    def update_stats(self, x_mean, x_std):
        """
        Update running mean and std with exponential moving average.

        Args:
            x_mean (Tensor): Batch mean, shape (d,)
            x_std (Tensor): Batch std, shape (d,)
        """
        self.x_mean = self.x_mean * self.forget + x_mean * (1 - self.forget)
        self.x_std = self.x_std * self.forget + x_std * (1 - self.forget)

def MLP(di, do, s=32, layers=4, activation=nn.Tanh, dropout=0.0):

    hidden_dim = di * s
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")

    def hidden_block(in_features):
        block = [nn.Linear(in_features, hidden_dim), activation()]
        if dropout > 0.0:
            block.append(nn.Dropout(p=dropout))
        return block

    net = hidden_block(di)
    for _ in range(layers - 1):
        net += hidden_block(hidden_dim)
    net += [nn.Linear(hidden_dim, do)]
    return nn.Sequential(*net)

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def softplus(x):
    return F.softplus(x)

def loglik_normal(x1: torch.Tensor, x2: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x1: (..., dx)
        x2: (..., dx)
        sig: (..., dx)
    """
    loglik = Normal(x1, sig).log_prob(x2)
    return loglik

def loglik_mvn_chol(y, y_pred, Py):
    """
    Args:
        y:       (..., dy)
        y_pred:  (..., dy)
        Py:      (..., dy, dy) - covariance matrix
    Returns:
        loglik: (...,)
    """
    dy = y.shape[-1]
    diff = y - y_pred  # (..., dy)

    # Cholesky decomposition for numerical stability
    L = torch.linalg.cholesky(Py)  # (..., dy, dy)

    # Solve for (y - mu)^T Σ^{-1} (y - mu) via Cholesky
    # First solve Σ X = (y - mu) ----> X = Σ^{-1} (y - mu)
    # Then solve diff^T X
    solve = torch.cholesky_solve(diff.unsqueeze(-1), L)  # (..., dy, 1) 计算的是(LL^T)^{-1} diff
    maha = torch.matmul(diff.unsqueeze(-2), solve).squeeze(-1).squeeze(-1)  # (..., ) 只乘最后两维

    # Log determinant: log|Σ| = 2 * sum(log(diag(L)))
    # 利用下三角矩阵性质计算秩
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # (...)

    log_likelihood = -0.5 * (logdet + maha)
    return log_likelihood  # shape: (..., )

def loglik_mvn(y, y_pred, A, U, C, V=None, flag_chol=True):
    """
    Args:
        y:       (..., dy)
        y_pred:  (..., dy)
        A:      (..., dy, dy) - covariance matrix
        U:      (..., dy, m) - low rank transfer matrix
        C:      (..., m, m) - low rank kernel matrix
        V:      (..., m, dy) - low rank transfer matrix
        Py = A + U C V
    Returns:
        loglik: (...,)
    """
    # 数据处理
    diff = y - y_pred  # (..., dy)
    if V is None:
        V = U.transpose(-2, -1)
    sigma2 = A.diagonal(dim1=-2, dim2=-1)  # (B, n)
    A_inv = torch.diag_embed(1.0 / sigma2)
    C_inv = inv_by_solve(C)
    C_inv = 0.5 * (C_inv + C_inv.transpose(-1, -2))


    # Solve for (y - mu)^T Σ^{-1} (y - mu)
    # 使用伍德伯里公式计算协方差矩阵逆
    # (A + UCV)^(-1) = A^{-1} - A^{-1}U middle^{-1} V A^{-1}
    # middle = C^{-1} + V A^{-1} U
    middle = C_inv + V @ A_inv @ U
    middle = 0.5 * (middle + middle.transpose(-1, -2))
    if not flag_chol:
        # 整体进行伍德伯里公式求逆
        inv_Py = A_inv - A_inv @ U @ torch.linalg.solve(middle, V @ A_inv)
        maha = torch.matmul(diff.unsqueeze(-2), torch.matmul(inv_Py, diff.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    else:
        # 逐项拆开，对middle运用cholesky分解计算
        L_mid = safe_cholesky(C_inv, V, A_inv, U)
        t = V @ A_inv @ diff.unsqueeze(-1)
        mid = torch.linalg.solve_triangular(L_mid, t, upper=False)
        maha1 = diff.unsqueeze(-2) @ A_inv @ diff.unsqueeze(-1)
        maha2 = mid.transpose(-2, -1) @ mid
        maha = (maha1 - maha2).squeeze(-1).squeeze(-1)


    # Log determinant: log|Σ| = log|A + U C V|=log|A| + log|C| + log|C^-1 + V A^-1 U|
    sign1 = torch.sign(sigma2).prod(dim=-1)
    logabsdet1 = torch.log(sigma2.abs()).sum(dim=-1)
    sign2, logabsdet2 = torch.linalg.slogdet(C)

    if not flag_chol:
        # 直接计算mid的行列式对数
        sign3, logabsdet3 = torch.linalg.slogdet(middle)
    else:
        # 利用cholesky分解结果得到行列式对数
        diag_L = torch.diagonal(L_mid, dim1=-2, dim2=-1)
        logabsdet3 = 2.0 * torch.sum(torch.log(diag_L), dim=-1)
        sign3 = torch.sign(diag_L).prod(dim=-1) ** 2

    logdet = logabsdet1 + logabsdet2 + logabsdet3

    # 先把 sign1、sign2 广播到与 sign3 相同形状
    sign1 = torch.broadcast_to(sign1, sign3.shape)
    sign2 = torch.broadcast_to(sign2, sign3.shape)

    # 堆叠成 (3, B)
    sign = torch.stack([sign1, sign2, sign3], dim=0)

    mask_sign = sign < 0

    if mask_sign.any():
        idx = torch.where(mask_sign)
        print(
            f"Warning: determinant |Σ| is non-positive in function loglike_mvn, "
            f"rows={idx[0].tolist()}, batch_index={idx[1].tolist()}"
        )

    log_likelihood = -0.5 * (logdet + maha)
    return log_likelihood  # shape: (..., )

def _hessian_diag_exact(loss, params):
    params = list(params())

    param_values = [p for _, p in params]

    grads = torch.autograd.grad(
        loss,
        param_values,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    hess_diag = []

    for param, grad in zip(param_values, grads):

        if grad is None or not grad.requires_grad:
            hess_diag.append(torch.zeros_like(param))
            continue

        grad_flat = grad.reshape(-1)

        diag_flat = []

        for i in range(grad_flat.numel()):
            second_grad = torch.autograd.grad(
                grad_flat[i],
                param,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if second_grad is None:
                diag_flat.append(torch.zeros((), dtype=param.dtype,device=param.device))
            else:
                diag_flat.append(second_grad.reshape(-1)[i])

        hess_diag.append(
            torch.stack(diag_flat)
            .view_as(param)
            .detach()
        )

    return hess_diag

def _hessian_diag_hutchinson(loss, params, n_samples):
        params = list(params())
        param_values = [param for _, param in params]
        grads = torch.autograd.grad(
            loss,
            param_values,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        hess_diag = {
            name: torch.zeros_like(param)
            for name, param in params
        }
        active = [
            (name, param, grad)
            for (name, param), grad in zip(params, grads)
            if grad is not None and grad.requires_grad
        ]
        if not active:
            return hess_diag

        for _ in range(n_samples):
            vectors = [
                torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)
                for _, param, _ in active
            ]
            grad_dot_v = sum(
                (grad * vector).sum()
                for (_, _, grad), vector in zip(active, vectors)
            )
            hvps = torch.autograd.grad(
                grad_dot_v,
                [param for _, param, _ in active],
                retain_graph=True,
                allow_unused=True,
            )

            for (name, _, _), vector, hvp in zip(active, vectors, hvps):
                if hvp is not None:
                    hess_diag[name] = hess_diag[name] + vector * hvp

        for name in hess_diag:
            hess_diag[name] = (hess_diag[name] / n_samples).detach()

        return hess_diag

def safe_cholesky(C_inv, V, A_inv, U, name="middle"):
    middle = C_inv + V @ A_inv @ U
    middle = 0.5 * (middle + middle.transpose(-1, -2))

    try:
        L = torch.linalg.cholesky(middle)
        return L
    except RuntimeError as e:
        print(f"[Cholesky ERROR] Matrix '{name}' is not SPD")

        # ===== 基本信息 =====
        print(f"shape: {middle.shape}")
        print(f"dtype: {middle.dtype}, device: {middle.device}")

        # ===== 对称性检查 =====
        sym_err = torch.norm(middle - middle.transpose(-1, -2))
        print(f"symmetry error ||A - A^T|| = {sym_err.item():.3e}")

        # ===== 特征值检查（最关键）=====
        try:
            eigvals = torch.linalg.eigvalsh(middle)
            eigvals_C_inv = torch.linalg.eigvalsh(C_inv)
            eigvals_VAU = torch.linalg.eigvalsh(V @ A_inv @ U)
            print(f"middle min eigenvalue = {eigvals.min().item():.3e}")
            print(f"middle max eigenvalue = {eigvals.max().item():.3e}")
            print(f"C_inv min eigenvalue = {eigvals_C_inv.min().item():.3e}")
            print(f"C_inv max eigenvalue = {eigvals_C_inv.max().item():.3e}")
            print(f"VAU min eigenvalue = {eigvals_VAU.min().item():.3e}")
            print(f"VAU max eigenvalue = {eigvals_VAU.max().item():.3e}")
        except Exception as eig_e:
            print("Eigenvalue computation failed:", eig_e)

        # ===== 是否含 NaN / Inf =====
        if torch.isnan(middle).any():
            print("Matrix contains NaN")
        if torch.isinf(middle).any():
            print("Matrix contains Inf")

        C_inv = C_inv.cpu().detach().numpy()
        A_inv = A_inv.cpu().detach().numpy()
        V = V.cpu().detach().numpy()
        U = U.cpu().detach().numpy()

        # ===== 抛出异常（保留原始错误）=====
        raise e

def inv_by_solve(A: torch.Tensor, flag_triangle=False, upper=False) -> torch.Tensor:

    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    I = I.expand(*A.shape[:-2], n, n)

    # A^{-1}
    if flag_triangle:
        A_inv = torch.linalg.solve_triangular(A, I, upper=upper)
    else:
        A_inv = torch.linalg.solve(A, I)

    return A_inv

def sparse_metric(x, eps=1e-5):
    """
    Args:
        x: (..., dx)
    Returns:
        (...,)
    """
    n = x.shape[-1]
    l1 = torch.norm(x, p=1, dim=-1)
    l2 = torch.norm(x, p=2, dim=-1)
    return l1 / (l2 + eps) * n**(-0.5)

def legendre_poly(x, n):
    """
    Parameters:
        x: Tensor of shape (..., dx)
        n: Number of Legendre polynomials to compute

    Returns:
        Tensor of shape (..., n), applied per feature
        (if dx > 1, it returns a list of tensors)
    """
    *batch_shape, dx = x.shape
    x = x.view(-1, dx)  # reshape to (N, dx)
    N = x.shape[0]

    results = []

    for d in range(dx):
        xd = x[:, d]  # shape (N,)
        P = torch.zeros(N, n, dtype=xd.dtype, device=xd.device)

        if n > 0:
            P[:, 0] = 1.0
        if n > 1:
            P[:, 1] = xd

        for k in range(2, n):
            P[:, k] = ((2 * k - 1) * xd * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

        results.append(P.view(*batch_shape, n))  # shape (..., n)

    if dx == 1:
        return results[0]  # shape (..., n)
    else:
        return results      # list of dx tensors, each of shape (..., n)

def ToTensor(x: Union[ndarray, List, Tensor], device=None, dtype=torch.float32):
    """to tensor"""
    if x is None:
        return x
    else:
        if isinstance(x, torch.Tensor):
            y = x
        else:
            y = torch.tensor(x)

        if dtype is not None:
            y = y.to(dtype)

        if device is not None:
            return y.to(device)

        return y

def Torch2Np(x):
    return x.detach().to('cpu').numpy()

def get_MSE(y, y_pred, dims_mean=None):
    e2 = (y - y_pred)**2
    if dims_mean is not None:
        mse = e2.mean()
    else:
        mse = e2.mean(dim=dims_mean)
    return mse

def AddDim(x, n, dim=0):
    for i in range(n):
        x = x.unsqueeze(dim)
    return x

class ParamSoftPlus(nn.Module):
    def __init__(self, x, is_train=True):
        super().__init__()

        self.x_raw = nn.Parameter(inv_softplus(x), requires_grad=is_train)

    def forward(self):
        return softplus(self.x_raw)


class ParamVarianceTree(nn.Module):
    """
    Trainable positive variances with the same tensor shapes as a module's parameters.
    """
    def __init__(self, module: nn.Module, init_var=1e-4, is_train=True):
        super().__init__()

        self.names = []
        self.raw_vars = nn.ParameterList()

        for name, param in module.named_parameters():
            init = torch.full_like(param.detach(), float(init_var))
            self.names.append(name)
            self.raw_vars.append(nn.Parameter(inv_softplus(init), requires_grad=is_train))

    def forward(self) -> OrderedDictType[str, Tensor]:
        return OrderedDict(
            (name, softplus(raw))
            for name, raw in zip(self.names, self.raw_vars)
        )

    def named_variances(self):
        return self.forward().items()

    def variance_for(self, name: str) -> Tensor:
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown variance parameter: {name}") from exc
        return softplus(self.raw_vars[index])

# 半正定矩阵生成
class ParamPSDMat(nn.Module):
    def __init__(self, x, is_train=True):
        super().__init__()

        self.L_raw = nn.Parameter(torch.linalg.cholesky(x), requires_grad=is_train)
        self.type = "PSD"

    def forward(self):
        return self.L @ self.L.transpose(-1, -2)

    @property
    def L(self):
        return torch.tril(self.L_raw)

class ParamPDMat(nn.Module):
    def __init__(self, x, is_train=True, eps=1e-5):
        super().__init__()

        self.L_raw = nn.Parameter(torch.linalg.cholesky(x), requires_grad=is_train)
        self.type = "PD"

        In = torch.eye(self.L_raw.shape[-1], dtype=self.L_raw.dtype)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("In", In)

    def forward(self):
        return self.L @ self.L.transpose(-1, -2) + self.eps * self.In

    @property
    def L(self):
        return torch.tril(self.L_raw)

# 对角矩阵生成
class ParamDiagMat(nn.Module):
    def __init__(self, x, is_train=True, eps=0.0):
        super().__init__()

        # x: 初始对角（可以是矩阵或向量）
        if x.ndim == 2:
            x = torch.diagonal(x)

        # 反 softplus 初始化（保证初值一致）
        self.raw = nn.Parameter(
            torch.log(torch.exp(x) - 1.0),
            requires_grad=is_train
        )

        self.register_buffer("eps", torch.tensor(eps))

        self.type = "Diag"

    def forward(self):
        return torch.diag(self.diag)

    @property
    def diag(self):
        return F.softplus(self.raw) + self.eps


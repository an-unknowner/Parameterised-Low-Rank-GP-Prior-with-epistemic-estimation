from abc import abstractmethod
from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.func import functional_call
from torch.xpu import device
from tqdm import tqdm
import numpy as np

from .kernel import MultiTaskRBFKer
from .ker_net import MlpRBFKer, TfKer, BfnKer
from .utils import MLP, loglik_mvn, ParamSoftPlus, ParamVarianceTree, inv_softplus
import utils

class BasePriorLearn(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args

        self.dx = self.args.dx
        self.dy = self.args.dy
        self.dz = self.args.dz

        self.stdM_param = ParamSoftPlus(self.args.stdM, is_train=False)
        self.register_buffer("x_center", torch.zeros(self.dx))
        self.register_buffer("x_scale", torch.ones(self.dx))
        self.register_buffer("y_center", torch.zeros(self.dy))
        self.register_buffer("y_scale", torch.ones(self.dy))
        self.init_var = getattr(self.args, "prior_var_pho", 1e2)

        self.init_prior()

        self.to(args.device)

    def init_prior(self):
        self.mean = MLP(
            di=self.dx, do=self.dy, s=32, activation=nn.GELU)
        self.init_ker()

    @abstractmethod
    def init_ker(self):
        pass

    @property
    def stdM(self):
        return self.stdM_param().view(-1)

    @property
    def stdM_physical(self):
        scale = self.y_scale if hasattr(self, "y_scale") else 1.0
        scale = scale if self.args.flag_normalize else 1.0
        return self.stdM * scale

    def set_normalization(self, x_center, x_scale, y_center, y_scale):
        """Set active normalization coefficients and matching noise units."""
        device, dtype = self.x_center.device, self.x_center.dtype
        if self.args.flag_normalize:
            active_x_center = torch.as_tensor(
                x_center, device=device, dtype=dtype).reshape(-1)
            active_x_scale = torch.as_tensor(
                x_scale, device=device, dtype=dtype).reshape(-1).clamp_min(1e-6)
            active_y_center = torch.as_tensor(
                y_center, device=device, dtype=dtype).reshape(-1)
            active_y_scale = torch.as_tensor(
                y_scale, device=device, dtype=dtype).reshape(-1).clamp_min(1e-6)
        else:
            # Identity coefficients make every normalization and inverse-
            # normalization path a no-op, including sampled predictions.
            active_x_center = torch.zeros_like(self.x_center)
            active_x_scale = torch.ones_like(self.x_scale)
            active_y_center = torch.zeros_like(self.y_center)
            active_y_scale = torch.ones_like(self.y_scale)

        self.x_center.copy_(active_x_center)
        self.x_scale.copy_(active_x_scale)
        self.y_center.copy_(active_y_center)
        self.y_scale.copy_(active_y_scale)
        noise = (
            torch.as_tensor(self.args.stdM, device=device, dtype=dtype).reshape(-1)
            / self.y_scale
        ).clamp_min(1e-8)
        with torch.no_grad():
            self.stdM_param.x_raw.copy_(inv_softplus(noise))

    def normalize_x(self, x):
        if not hasattr(self, "x_center") or not hasattr(self, "x_scale"):
            return x
        return (x - self.x_center) / self.x_scale

    def normalize_y(self, y):
        if not hasattr(self, "y_center") or not hasattr(self, "y_scale"):
            return y
        return (y - self.y_center) / self.y_scale

    def predict_mean(self, x):
        value = self.mean(self.normalize_x(x))
        if not hasattr(self, "y_scale") or not hasattr(self, "y_center"):
            return value
        return value * self.y_scale + self.y_center

    def predict_phi(self, x):
        value = self.ker.phi(self.normalize_x(x))
        if not hasattr(self, "y_scale"):
            return value
        return value * self.y_scale.reshape(-1)[0]

    def predict_cov(self, x):
        phi = self.predict_phi(x)
        return phi @ self.ker.P0 @ phi.transpose(-1, -2)

    @property
    def ker_param_var(self):
        return self.ker_var()

class ExactPL(BasePriorLearn):
    """Exact likelihood + pathwise derivative for independent Gaussian parameters"""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # Stage 2 deliberately uses a zero-mean parameter prior. Stage-1
        # information is retained by the function-output guide, not p(w)'s
        # centre.
        prior_mean = 0.0

        prior_rho = torch.tensor(
            getattr(self.args, "prior_std_rho", 0.2),
            dtype=torch.float32
        )
        prior_std = F.softplus(prior_rho).item()

        self.mean_named_params = list(self.mean.named_parameters())
        self.phi_named_params = list(self.ker.phi.named_parameters())

        self.mean_param_names = [n for n, _ in self.mean_named_params]
        self.phi_param_names = [n for n, _ in self.phi_named_params]

        self.net_param = (
            [p for _, p in self.mean_named_params]
            + [p for _, p in self.phi_named_params]
        )

        self.n_mean_param = len(self.mean_named_params)

        # q(w) 的 std 参数，rho -> std
        self.rho_param = nn.ParameterList([
            nn.Parameter(
                torch.randn_like(p) * args.init_rho_noise + args.init_std_rho
            )
            for p in self.net_param
        ])

        # 与 mean 网络对应的 rho
        self.mean_rho_param = nn.ParameterList(
            list(self.rho_param[:self.n_mean_param])
        )

        # 与 phi 网络对应的 rho
        self.phi_rho_param = nn.ParameterList(
            list(self.rho_param[self.n_mean_param:])
        )

        # p(w) 的先验均值
        self.prior_mean = [
            torch.full_like(p, prior_mean)
            for p in self.net_param
        ]

        # p(w) 的先验标准差，不设为 Parameter
        self.prior_std = [
            torch.full_like(p, prior_std)
            for p in self.net_param
        ]

        # Frozen deterministic networks used by the Stage-2 output guides.
        # They are created only after Stage-1 has finished.  These references
        # do not alter p(w), whose parameter means remain zero above.
        self.pretrained_mean_reference = None
        self.pretrained_phi_reference = None

    def init_ker(self):
        ker_map = {
            'mlp_rbf': MlpRBFKer,
            'tf': TfKer,
            'bfn': BfnKer,
        }
        if self.args.ker_net not in ker_map:
            raise ValueError(f"Unknown model name: {self.args.ker_net}")
        self.ker = ker_map[self.args.ker_net](self.args)

    def get_param_std(self):
        return [F.softplus(rho) for rho in self.rho_param]

    @torch.no_grad()
    def set_pretrained_network_references(self):
        """Freeze both Stage-1 network functions as Stage-2 guide targets."""
        self.pretrained_mean_reference = copy.deepcopy(self.mean)
        self.pretrained_mean_reference.requires_grad_(False)
        self.pretrained_mean_reference.eval()
        self.pretrained_phi_reference = copy.deepcopy(self.ker.phi)
        self.pretrained_phi_reference.requires_grad_(False)
        self.pretrained_phi_reference.eval()

    @torch.no_grad()
    def set_pretrained_mean_reference(self):
        """Backward-compatible alias that now freezes both networks."""
        self.set_pretrained_network_references()

    def mean_output_guide_loss(self, x):
        """Keep the posterior-mean network output near its Stage-1 function."""
        if self.pretrained_mean_reference is None:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x_norm = self.normalize_x(x)
        # Compare deterministic functions: Dropout noise must not itself be
        # penalized as drift from the pretrained result.
        was_training = self.mean.training
        self.mean.eval()
        current = self.mean(x_norm)
        self.mean.train(was_training)
        self.pretrained_mean_reference.eval()
        with torch.no_grad():
            target = self.pretrained_mean_reference(x_norm)
        return F.mse_loss(current, target)

    def phi_output_guide_loss(self, x):
        """Guide every phi output toward its frozen Stage-1 function value."""
        if self.pretrained_phi_reference is None:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        x_norm = self.normalize_x(x)
        was_training = self.ker.phi.training
        self.ker.phi.eval()
        current = self.ker.phi(x_norm)
        self.ker.phi.train(was_training)
        self.pretrained_phi_reference.eval()
        with torch.no_grad():
            target = self.pretrained_phi_reference(x_norm)

        # Reduce input/batch dimensions first, leaving one loss per phi output;
        # the final mean gives every basis output equal guide weight.
        reduce_dims = tuple(range(current.ndim - 1))
        per_phi_loss = (current - target).square().mean(dim=reduce_dims)
        return per_phi_loss.mean()

    def sample_net_param(self):
        """
        Pathwise reparameterization:
            w = mu + std * eps
        得到新的网络参数list
        """
        param_std = self.get_param_std()

        sampled_param = [
            mu + std * torch.randn_like(std)
            for mu, std in zip(self.net_param, param_std)
        ]

        return sampled_param

    def split_sampled_param(self, sampled_param):
        """
        分割采样结果为mean参数和phi参数
        """

        sampled_mean_param = sampled_param[:self.n_mean_param]
        sampled_phi_param = sampled_param[self.n_mean_param:]

        mean_param_dict = {
            name: p
            for name, p in zip(self.mean_param_names, sampled_mean_param)
        }

        phi_param_dict = {
            name: p
            for name, p in zip(self.phi_param_names, sampled_phi_param)
        }

        return mean_param_dict, phi_param_dict

    def loglik_sampled(self, x, y, sampled_param):
        """
        用采样后的网络参数计算 log likelihood。
        这里没有直接覆盖 self.mean / self.ker.phi 的参数，
        而是用 functional_call 保持计算图可导。
        """

        n, dx, dy = x.shape[-2], x.shape[-1], y.shape[-1]

        mean_param_dict, phi_param_dict = self.split_sampled_param(sampled_param)

        mean_state = dict(self.mean.named_buffers())
        mean_state.update(mean_param_dict)

        phi_state = dict(self.ker.phi.named_buffers())
        phi_state.update(phi_param_dict)
        # 将参数调整为网络同结构，便于后续进入网络调用

        x_norm = self.normalize_x(x)
        y_norm = self.normalize_y(y)
        y_pred = functional_call(self.mean, mean_state, (x_norm,))
        phi = functional_call(self.ker.phi, phi_state, (x_norm,))

        Y = y_norm.reshape(*y.shape[:-2], n * dy)
        Y_pred = y_pred.reshape(*y.shape[:-2], n * dy)

        P0 = self.ker.P0

        # TODO: check stdM
        cov_noise = torch.diag_embed(self.stdM)
        Pnoise = torch.kron(
            torch.eye(n, device=x.device, dtype=x.dtype),
            cov_noise
        )

        l = loglik_mvn(Y, Y_pred, Pnoise, phi, P0, flag_chol=True)

        return l

    def kl_divergence(self):
        """
        KL(q(w)||p(w))，其中 q 和 p 都是独立高斯：

            q(w_i) = N(mu_i, sigma_i^2)
            p(w_i) = N(mu0_i, sigma0_i^2)

        KL = log(sigma0/sigma)
             + (sigma^2 + (mu-mu0)^2)/(2 sigma0^2)
             - 1/2
        """

        param_std = self.get_param_std()

        kl = 0.0

        for mu, std, mu0, std0 in zip(
            self.net_param,
            param_std,
            self.prior_mean,
            self.prior_std
        ):
            kl_i = (
                torch.log(std0 / std)
                + (std ** 2 + (mu - mu0) ** 2) / (2.0 * std0 ** 2)
            ).sum()

            kl = kl + kl_i

        return kl

    def loss(self, x, y, use_sampling=False, include_kl=True,
             include_mean_guide=True, include_phi_guide=True):
        """
        MC 估计 ELBO 的负值：

            loss = E_q[-log p(y|w)] + KL(q(w)||p(w))

        通过 w = mu + sigma * eps，
        likelihood 项同时对 mu 和 sigma 可导。

        use_sampling = False:
        只使用均值参数 w = mu
        likelihood 项只优化 mu
        sigma / rho 只通过 KL 项更新
        """

        mc_samples = getattr(self.args, "mc_samples", 1)

        if use_sampling:
            loss_nll = 0.0

            for _ in range(mc_samples):
                sampled_param = self.sample_net_param()
                loglik = self.loglik_sampled(x, y, sampled_param)
                loss_nll = loss_nll - loglik.mean()

            loss_nll = loss_nll / mc_samples

        else:
            # 只使用均值参数，不采样
            mean_param = list(self.net_param)

            loglik = self.loglik_sampled(x, y, mean_param)

            loss_nll = -loglik.mean()

        if not include_kl:
            zero = torch.zeros((), device=loss_nll.device, dtype=loss_nll.dtype)
            return loss_nll, loss_nll, zero

        loss_kl = self.kl_divergence() / self.args.n_train_traj    # 除以样本总量，归一化KL散度

        beta_kl = getattr(self.args, "beta_kl", 1.0)

        loss_guide = torch.zeros_like(loss_nll)
        if include_mean_guide:
            loss_guide = loss_guide + (
                getattr(self.args, "lambda_mean_guide", 1.0)
                * self.mean_output_guide_loss(x))
        if include_phi_guide:
            loss_guide = loss_guide + (
                getattr(self.args, "lambda_phi_guide", 1.0)
                * self.phi_output_guide_loss(x))

        return (loss_nll + beta_kl * loss_kl + loss_guide,
                loss_nll, loss_kl * beta_kl)

    def res_check(self, x, y):
        """
        测试时建议使用参数均值，而不是随机采样。
        """

        sampled_param = self.net_param
        loss = -self.loglik_sampled(x, y, sampled_param).mean()

        return loss

    def forward_sampled(self, x, sampled_param=None):
        """
        给定一组采样网络参数，计算：
            y_pred : (..., n, dy)
            phi    : (..., n*dy, r) 或者与你 loglik_mvn 匹配的形状

        注意：
        x 用于 WingRock 时一般为 (..., n, 2)
        """

        if sampled_param is None:
            sampled_param = self.net_param

        mean_param_dict, phi_param_dict = self.split_sampled_param(sampled_param)

        mean_state = dict(self.mean.named_buffers())
        mean_state.update(mean_param_dict)

        phi_state = dict(self.ker.phi.named_buffers())
        phi_state.update(phi_param_dict)

        x_norm = self.normalize_x(x)
        y_pred = functional_call(self.mean, mean_state, (x_norm,))
        phi = functional_call(self.ker.phi, phi_state, (x_norm,))
        if hasattr(self, "y_scale") and hasattr(self, "y_center"):
            y_pred = y_pred * self.y_scale + self.y_center
            phi = phi * self.y_scale.reshape(-1)[0]

        return y_pred, phi

    def cov_from_phi(self, phi):
        """
        由 phi 和 P0 计算单个采样网络对应的函数协方差：

            K = phi P0 phi^T

        返回：
            cov_func  : (..., n*dy, n*dy)
            cov_noise : (n*dy, n*dy)
            cov_total : (..., n*dy, n*dy)
        """

        n = phi.shape[-2]
        dy = self.stdM.numel()

        P0 = self.ker.P0

        cov_func = phi @ P0 @ phi.transpose(-1, -2)

        cov_noise_single = torch.diag_embed(self.stdM_physical ** 2)

        cov_noise = torch.kron(
            torch.eye(n, device=phi.device, dtype=phi.dtype),
            cov_noise_single
        )

        cov_total = cov_func + cov_noise

        return cov_func, cov_noise, cov_total

    @torch.no_grad()
    def predict_gmm(
            self,
            x,
            n_samples=50,
            include_noise=True,
            return_full_cov=True,
    ):
        """
        通过 q(w) 随机采样多个网络参数，形成 GMM 预测。

        参数
        ----
        x : Tensor
            WingRock 输入，推荐形状：
                (n, 2) 或 (B, n, 2)

        n_samples : int
            网络参数采样次数，即 GMM 分量数。

        include_noise : bool
            True  : total uncertainty 包含观测噪声
            False : total uncertainty 只包含函数不确定性

        use_mean_param : bool
            True  : 不采样，只重复使用参数均值，相当于退化 GMM
            False : 从 q(w) 中采样网络参数

        return_full_cov : bool
            True  : 返回完整协方差矩阵 (..., n*dy, n*dy)
            False : 只返回对角线方差 (..., n, dy)

        返回
        ----
        out : dict
            {
                "mean":       预测均值 (..., n, dy),
                "var_total":  总方差 (..., n, dy),
                "var_epi":    认知方差 (..., n, dy),
                "var_alea":   偶然方差 (..., n, dy),
                "cov_total":  可选，完整总协方差,
                "cov_epi":    可选，完整认知协方差,
                "cov_alea":   可选，完整偶然协方差,
                "component_mean": 每个分量均值 (M, ..., n, dy),
                "component_var":  每个分量方差 (M, ..., n, dy)
            }
        """

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=dtype, device=device)
        else:
            x = x.to(device=device, dtype=dtype)

        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, n, 2)


        n = x.shape[-2]
        batch_shape = x.shape[:-2]

        mean_list = []
        phi_list = []
        cov_func_list = []
        cov_total_list = []
        var_list = []

        for _ in range(n_samples):
            if n_samples == 1:
                sampled_param = list(self.net_param)
            else:
                sampled_param = self.sample_net_param()

            y_pred, phi = self.forward_sampled(x, sampled_param)

            dy = y_pred.shape[-1]
            dz = phi.shape[-1]

            y_vec = y_pred.reshape(*batch_shape, n * dy)

            if return_full_cov:
                cov_func, cov_noise, cov_total = self.cov_from_phi(phi)
                cov_used = cov_total if include_noise else cov_func
            else:
                # Pointwise variance is enough for grid plots.  Avoid forming
                # the dense N x N covariance matrix.
                P0 = self.ker.P0.to(device=phi.device, dtype=phi.dtype)
                var_used = torch.einsum("...ni,ij,...nj->...n", phi, P0, phi)
                if include_noise:
                    var_used = var_used + (self.stdM ** 2).reshape(-1)[0]

            mean_list.append(y_vec)
            phi_list.append(phi)
            if return_full_cov:
                cov_func_list.append(cov_func)
                cov_total_list.append(cov_used)
            else:
                var_list.append(var_used)

        means = torch.stack(mean_list, dim=0)  # (M, ..., n*dy)
        phis = torch.stack(phi_list, dim=0)     # (M, ..., n, dz)
        if not return_full_cov:
            component_var_flat = torch.stack(var_list, dim=0)
            y_mean_gmm = means.mean(dim=0)
            phi_mean_gmm = phis.mean(dim=0)
            phi_var_gmm = torch.var(phis, dim=0, correction=0)
            var_alea_flat = component_var_flat.mean(dim=0)
            var_epi_flat = torch.var(means, dim=0, correction=0)
            var_total_flat = var_alea_flat + var_epi_flat
            dy = y_mean_gmm.shape[-1] // n
            return {
                "mean": y_mean_gmm.reshape(*batch_shape, n, dy),
                "var_total": var_total_flat.reshape(*batch_shape, n, dy),
                "var_epi": var_epi_flat.reshape(*batch_shape, n, dy),
                "var_alea": var_alea_flat.reshape(*batch_shape, n, dy),
                "component_mean": means.reshape(n_samples, *batch_shape, n, dy),
                "component_phi": phis,
                "component_var": component_var_flat.reshape(
                    n_samples, *batch_shape, n, dy),
                "phi_mean": phi_mean_gmm,
                "phi_var": phi_var_gmm,
            }

        covs = torch.stack(cov_total_list, dim=0)  # (M, ..., n*dy, n*dy)

        y_mean_gmm = means.mean(dim=0)  # (..., n*dy)
        phi_mean_gmm = phis.mean(dim=0)
        phi_var_gmm = torch.var(phis, dim=0, correction=0)

        # aleatoric: E[Sigma_m]
        # cov_alea, _, _ = self.cov_from_phi(phi_mean_gmm)
        # cov_var, _, _ = self.cov_from_phi(phi_var_gmm)

        cov_alea = covs.mean(dim=0)
        cov_var = covs.var(dim=0, unbiased=False)


        cov_std = torch.sqrt(torch.clamp(cov_var, min=0.0))

        # epistemic: Cov(mu_m)
        mean_outer = means.unsqueeze(-1) @ means.unsqueeze(-2)
        cov_epi = mean_outer.mean(dim=0) - y_mean_gmm.unsqueeze(-1) @ y_mean_gmm.unsqueeze(-2)

        # total = aleatoric + epistemic
        cov_total = cov_alea + cov_epi

        dy = y_mean_gmm.shape[-1] // n

        mean_out = y_mean_gmm.reshape(*batch_shape, n, dy)

        var_alea = torch.diagonal(cov_alea, dim1=-2, dim2=-1).reshape(*batch_shape, n, dy)
        var_epi = torch.diagonal(cov_epi, dim1=-2, dim2=-1).reshape(*batch_shape, n, dy)
        var_total = torch.diagonal(cov_total, dim1=-2, dim2=-1).reshape(*batch_shape, n, dy)

        component_mean = means.reshape(n_samples, *batch_shape, n, dy)
        component_phi = phis.reshape(n_samples, *batch_shape, n, dz)
        component_var = torch.diagonal(covs, dim1=-2, dim2=-1).reshape(
            n_samples, *batch_shape, n, dy
        )

        out = {
            "mean": mean_out,
            "var_total": var_total,
            "var_epi": var_epi,
            "var_alea": var_alea,
            "component_mean": component_mean,
            "component_phi": component_phi,
            "component_var": component_var,
            "cov_var": cov_var,
            "cov_std": cov_std,
        }

        if return_full_cov:
            out["cov_total"] = cov_total
            out["cov_epi"] = cov_epi
            out["cov_alea"] = cov_alea

        return out

def PriorLearn(args, **kwargs):
    model_map = {
        'exact': ExactPL,
    }
    if args.model not in model_map:
        raise ValueError(f"Unknown model name: {args.model}")
    return model_map[args.model](args, **kwargs)


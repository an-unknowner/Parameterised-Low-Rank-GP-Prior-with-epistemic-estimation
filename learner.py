import argparse
import copy
import os
from types import SimpleNamespace
from typing import Any, Dict

from torch import Tensor
from numpy import ndarray

import numpy as np
import torch
from torch.func import jacrev, vmap, functional_call
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import utils, data, res
from gp_prior_learn.prior_learn import PriorLearn
from utils import Np2Tensor, Torch2Np


class Train(utils.ITrainAttrLog):
    def __init__(self, args):
        super().__init__(logpath=args.log_path)
        self.args = args

        dataset = utils.load_pickle(args.dataset_path)
        self.data = dataset
        self.data_train, self.data_test = dataset["train"], dataset["test"]

        # 统一转换为 tensor
        for key, value in self.data_train.items():
            self.data_train[key] = utils.Np2Tensor(x=self.data_train[key], device=args.device)
        for key, value in self.data_test.items():
            self.data_test[key] = utils.Np2Tensor(x=self.data_test[key], device=args.device)

        # 保证关键训练数据维度正确
        self.data_train["x"] = utils.ensure_3d(self.data_train["x"])
        self.data_train["y"] = utils.ensure_3d(self.data_train["y"])
        self.data_train["yn"] = utils.ensure_3d(self.data_train["yn"])

        self.data_test["x"] = utils.ensure_3d(self.data_test["x"])
        self.data_test["y"] = utils.ensure_3d(self.data_test["y"])
        self.data_test["yn"] = utils.ensure_3d(self.data_test["yn"])

        print("Train x shape:", tuple(self.data_train["x"].shape))
        print("Train yn shape:", tuple(self.data_train["yn"].shape))
        print("Test x shape:", tuple(self.data_test["x"].shape))
        print("Test yn shape:", tuple(self.data_test["yn"].shape))

        # 构造模型
        self.model = PriorLearn(args).to(args.device)
        self.train_x = self.data_train["x"]
        self.train_y = self.data_train["yn"]

        # 训练集标准化参数；保存在模型 buffer 中，预测时自动复用。
        reduce_dims_x = tuple(range(self.train_x.ndim - 1))
        reduce_dims_y = tuple(range(self.train_y.ndim - 1))

        self.x_center = self.train_x.mean(dim=reduce_dims_x, keepdim=True)
        self.x_scale = self.train_x.std(
            dim=reduce_dims_x, keepdim=True, unbiased=False).clamp_min(1e-6)
        self.y_center = self.train_y.mean(dim=reduce_dims_y, keepdim=True)
        self.y_scale = self.train_y.std(
            dim=reduce_dims_y, keepdim=True, unbiased=False).clamp_min(1e-6)
        self.model.set_normalization(
            self.x_center, self.x_scale, self.y_center, self.y_scale)

        # Log the coefficients actually used by the model. When normalization
        # is disabled these are identity coefficients (centres 0, scales 1).
        self.x_center = self.model.x_center.detach().clone()
        self.x_scale = self.model.x_scale.detach().clone()
        self.y_center = self.model.y_center.detach().clone()
        self.y_scale = self.model.y_scale.detach().clone()

        # 用于方差训练的批设置
        self.batch_size = getattr(args, "batch_size", 32)

        # 待优化参数及其优化器
        self.optim_pre_mean = torch.optim.Adam(
            list(self.model.mean.parameters()),
            lr=args.lr_pretrain,
            weight_decay=args.wd
        )
        self.optim_pre_ker = torch.optim.Adam(
            list(self.model.ker.parameters()),
            lr=args.lr_pretrain,
            weight_decay=args.wd
        )
        self.optim_mean = torch.optim.Adam(
            list(self.model.mean.parameters()) + list(self.model.mean_rho_param),
            lr=args.lr,
            weight_decay=args.wd
        )
        self.optim_ker = torch.optim.Adam(
            list(self.model.ker.parameters()) + list(self.model.phi_rho_param),
            lr=args.lr,
            weight_decay=args.wd
        )

        # 优化器学习率调度
        self.sched_pre_mean = utils.get_scheduler(
            self.optim_pre_mean,
            max(1, self.args.n_pre_mean * self.args.n_mean),
            self.args.lr_pretrain, type=args.scheduler_pretrain
        )
        self.sched_pre_ker = utils.get_scheduler(
            self.optim_pre_ker,
            max(1, self.args.n_pre_mean * self.args.n_cov),
            self.args.lr_pretrain, type=args.scheduler_pretrain
        )
        self.sched_mean = utils.get_scheduler(
            self.optim_mean, self.args.n_iters * self.args.n_mean, self.args.lr, type=args.scheduler
        )
        self.sched_ker = utils.get_scheduler(
            self.optim_ker, self.args.n_iters * self.args.n_cov, self.args.lr, type=args.scheduler
        )

        self.data_recorder = utils.DataRecorder()
        self.log_content = [
            'model', 'optim_mean', 'losses', 'losses_pretrain',
            'losses_mean', 'losses_phi', 'data', 'sched_mean',
            'data_recorder', "x_center", "x_scale", "y_center", "y_scale",
            "mean_pretrain_diagnostics", "pretrained_mean_checkpoint",
            "stage1_network_mean_parameters"
        ]
        self.add_names_log(self.log_content)

    def run(self):

        # Stage 1 uses Dropout as deterministic-network regularization.
        self.model.train()

        ### Stage 1: alternate deterministic MLE updates of the mean and basis
        # networks. Variational rho parameters are deliberately not optimized.
        self.losses_pretrain = []
        print(
            f"\n[Stage 1/2] Mean/basis MLE pretraining: "
            f"{self.args.n_pre_mean} outer iterations "
            f"(deterministic likelihood, alternating updates)")
        pbar_pretrain = tqdm(
            range(self.args.n_pre_mean),
            desc="mean/basis MLE pretraining",
            total=self.args.n_pre_mean,
            leave=True,
            position=0,
        )
        pretrain_param_ids = {
            id(p) for p in list(self.model.mean.parameters())
            + list(self.model.ker.parameters())
        }
        frozen = [
            (p, p.requires_grad) for p in self.model.parameters()
            if id(p) not in pretrain_param_ids
        ]
        for parameter, _ in frozen:
            parameter.requires_grad_(False)
        try:
            for _ in pbar_pretrain:
                for _ in range(self.args.n_mean):
                    mean_loss = self.optimize_step(
                        self.optim_pre_mean, self.sched_pre_mean,
                        use_sampling=False, sample_batch=False,
                        pretrain_mle=True)
                    self.losses_pretrain.append(mean_loss)
                for _ in range(self.args.n_cov):
                    phi_loss = self.optimize_step(
                        self.optim_pre_ker, self.sched_pre_ker,
                        use_sampling=False, sample_batch=False,
                        pretrain_mle=True)
                    self.losses_pretrain.append(phi_loss)
                pbar_pretrain.set_postfix(
                    mean_nll=mean_loss[0], phi_nll=phi_loss[0])
        finally:
            for parameter, requires_grad in frozen:
                parameter.requires_grad_(requires_grad)

        # Preserve the deterministic solution before ELBO training can change it.
        # Freeze its output function for the Stage-2 guide. The ELBO prior stays
        # zero-mean for both networks.
        self.mean_pretrain_diagnostics = self.evaluate_mean_pretraining()
        self.model.set_pretrained_network_references()
        self.pretrained_mean_checkpoint = self.build_pretrained_mean_checkpoint()
        self.stage1_network_mean_parameters = (
            self.collect_network_mean_parameters())

        # Keep the Stage-2 fields present in pretraining-only logs so existing
        # plotting/loading code can consume the log without special cases.
        self.losses = []
        self.losses_mean = []
        self.losses_phi = []

        if getattr(self.args, "pretrain_only", False):
            pretrain_log_path = getattr(
                self.args, "pretrain_log_path", self.args.log_path)
            self.save_log(name=pretrain_log_path)
            self.log = self.get_log()
            print(
                "Pretraining-only mode: Stage 1 log saved; "
                f"Stage 2 skipped. path: {pretrain_log_path}")
            return

        # Explicitly initialize both q(w_mean) and q(w_phi) centres from the
        # deterministic solution; p(w) remains zero-mean for both networks.
        self.initialize_stage2_networks_from_pretraining()

        ### Stage 2 still computes gradients and updates all ELBO parameters, but
        # eval mode disables Dropout in both the mean and basis networks. The
        # Bayesian parameter sampling remains active through use_sampling=True.
        self.model.eval()

        # Stage 2: alternate mean/phi updates using the configured BNN loss.
        print(
            f"\n[Stage 2/2] Joint training: "
            f"{self.args.n_iters} outer iterations")
        pbar_joint = tqdm(
            range(self.args.n_iters),
            desc="joint mean/phi training",
            total=self.args.n_iters,
            leave=True,
            position=0,
        )

        for _ in pbar_joint:
            for _ in range(self.args.n_mean):
                mean_loss = self.optimize_step(
                    self.optim_mean, self.sched_mean,
                    use_sampling=True, sample_batch=True)
                self.losses_mean.append(mean_loss)
            for _ in range(self.args.n_cov):
                phi_loss = self.optimize_step(
                    self.optim_ker, self.sched_ker,
                    use_sampling=True, sample_batch=True)
                self.losses_phi.append(phi_loss)

            # Compatibility field: one representative loss per outer cycle.
            self.losses.append(phi_loss)

            self.param_update()
            pbar_joint.set_postfix(
                mean=mean_loss[0],
                phi=phi_loss[0],
                stdM=self.stdM,
                ls=self.ls,
                var=self.var
            )

            # self.update_log_best(loss.item())

        self.save_log(name=self.args.log_path)
        self.log = self.get_log()

        print("training log saved. path: {}".format(self.args.log_path))

    @torch.no_grad()
    def evaluate_mean_pretraining(self):
        """Collect compact, reproducible diagnostics at the pretraining boundary."""
        self.model.eval()
        diagnostics: Dict[str, Any] = {
            "losses": np.asarray(self.losses_pretrain, dtype=np.float64),
        }

        for split, data_split in (
                ("train", self.data_train), ("test", self.data_test)):
            x = data_split["x"]
            y = data_split["yn"]
            pred = self.model.predict_mean(x)
            error = pred - y
            normalized_error = (
                self.model.normalize_y(pred) - self.model.normalize_y(y))
            diagnostics[f"{split}_mse_normalized"] = (
                normalized_error.square().mean().detach().cpu().item())
            diagnostics[f"{split}_mse_physical"] = (
                error.square().mean().detach().cpu().item())
            diagnostics[f"{split}_mae_physical"] = (
                error.abs().mean().detach().cpu().item())
        self.model.train()
        return diagnostics

    def build_pretrained_mean_checkpoint(self):
        """Build the deterministic Stage-1 network checkpoint in log.pkl."""
        return {
            "mean_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.model.mean.state_dict().items()
            },
            "mean_prior": [
                value.detach().cpu().clone()
                for value in self.model.prior_mean[:self.model.n_mean_param]
            ],
            "phi_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.model.ker.phi.state_dict().items()
            },
            "ker_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.model.ker.state_dict().items()
            },
            "phi_prior": [
                value.detach().cpu().clone()
                for value in self.model.prior_mean[self.model.n_mean_param:]
            ],
            "x_center": self.model.x_center.detach().cpu().clone(),
            "x_scale": self.model.x_scale.detach().cpu().clone(),
            "y_center": self.model.y_center.detach().cpu().clone(),
            "y_scale": self.model.y_scale.detach().cpu().clone(),
        }

    @torch.no_grad()
    def collect_network_mean_parameters(self):
        """Snapshot the parameter centres of the mean and phi networks."""
        return {
            "mean_values": np.concatenate([
                parameter.detach().cpu().numpy().reshape(-1)
                for parameter in self.model.mean.parameters()
            ]),
            "phi_values": np.concatenate([
                parameter.detach().cpu().numpy().reshape(-1)
                for parameter in self.model.ker.phi.parameters()
            ]),
        }

    @torch.no_grad()
    def initialize_stage2_networks_from_pretraining(self):
        """Restore both variational centres from the Stage-1 snapshot."""
        checkpoint = self.pretrained_mean_checkpoint
        self.model.mean.load_state_dict(
            checkpoint["mean_state_dict"], strict=True)
        if "ker_state_dict" in checkpoint:
            self.model.ker.load_state_dict(
                checkpoint["ker_state_dict"], strict=True)
        else:
            self.model.ker.phi.load_state_dict(
                checkpoint["phi_state_dict"], strict=True)

        # Stage-2 optimizers are intentionally separate from the pretraining
        # optimizers. Clear both states at the phase boundary.
        self.optim_mean.state.clear()
        self.optim_ker.state.clear()

    @torch.no_grad()
    def initialize_stage2_mean_from_pretraining(self):
        """Backward-compatible alias for restoring both Stage-2 networks."""
        self.initialize_stage2_networks_from_pretraining()

    def param_update(self):
        self.stdM = utils.Torch2Np(self.model.stdM)
        self.ls = None
        self.var = None

    @torch.no_grad()
    def eval_test_loss(self):
        was_training = self.model.training
        self.model.eval()
        test_loss = -self.model.loglik(self.data_test["x"], self.data_test["yn"]).mean()
        self.model.train(was_training)
        return test_loss.item()

    def optimize_step(self, optimizer, scheduler, use_sampling=False,
                      sample_batch=True, use_mse=False, pretrain_mle=False):

        if sample_batch and self.train_x.shape[0] > self.args.batch_size:
            x_batch, y_batch = self.get_batch(use_sampling=True)
        else:
            x_batch, y_batch = self.get_batch(use_sampling=False)

        optimizer.zero_grad()

        if use_mse:
            y_pred = self.model.mean(self.model.normalize_x(x_batch))
            loss = torch.mean(
                (y_pred - self.model.normalize_y(y_batch)) ** 2)
            loss_nll = loss
            loss_kl = torch.zeros_like(loss)
        elif pretrain_mle:
            loss, loss_nll, loss_kl = self.model.loss(
                x_batch, y_batch, use_sampling=use_sampling, include_kl=False,
                include_mean_guide=False, include_phi_guide=False)
        else:
            loss, loss_nll, loss_kl = self.model.loss(
                x_batch, y_batch, use_sampling=use_sampling, include_kl=True,
                include_mean_guide=True, include_phi_guide=True)
        loss.backward()

        optimizer.step()
        if self.args.isSchdStep:
            if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

        return [loss.item(), loss_nll.item(), loss_kl.item()]

    def get_batch(self, use_sampling=False):
        """
        根据要求返回训练数据
        :param use_sampling: False时返回全量训练数据
                             True时从训练数据中抽取batch，有放回，和dataloader不同
        """

        if not use_sampling:
            return self.train_x, self.train_y

        else:
            idx = torch.randint(
                0,
                self.data_train["x"].shape[0],
                (self.batch_size,),
                device=self.train_x.device
            )

            return self.train_x[idx], self.train_y[idx]

class Predict:
    def __init__(self, args):
        self.args = args
        train_log = utils.load_pickle(args.log_path)
        self.dataset = utils.load_pickle(args.dataset_path)

        self.model = train_log["model"].to(args.device)
        # Prediction is deterministic with respect to Dropout.  Bayesian
        # parameter uncertainty is still represented by predict_gmm sampling.
        self.model.eval()
        self.ker = self.model.ker

    def traj_identify(self, id, data="train", flag_compare=False):
        '''对整条轨迹进行参数条件高斯预测
        取data中第id条轨迹 建模 y = m(x) + phi(x)θ + w
        其中w为测量噪声，利用联合分布 # [y] ~ N( [ m(x)]  [ φ P φ^T + Σ_n ⊗ I_n      φ P] )
                                  [θ]    ( [  0  ], [P φ^T                    P   ] )
        进行条件高斯分布预测，采用伍德伯里公式求逆减少计算量
        '''

        # 读取数据
        data = self.dataset[data]
        device = self.args.device
        traj_x = Np2Tensor(data["x"][id], device=device)
        traj_y = Np2Tensor(data["y"][id], device=device)
        traj_yn = Np2Tensor(data["yn"][id], device=device)

        # 高斯过程参数批计算
        if not flag_compare:
            P = self.ker.P0.to(device)
        else:
            P = torch.eye(self.ker.P0.shape[-1], device=device, dtype=self.ker.P0.dtype) * 1000
        y_mean = self.model.predict_mean(traj_x)
        phi = self.model.predict_phi(traj_x)
        n = traj_x.shape[-2]
        Sigma_noise = self.args.stdM * self.args.stdM * torch.eye(n)
        Sigma_noise = Sigma_noise.to(device)

        # 条件高斯预测
        Sigma21 = P @ phi.transpose(-1, -2)
        inv_Sigma11 = utils.woodbury_inverse_torch(Sigma_noise, phi, P)

        theta_m = Sigma21 @ inv_Sigma11 @ (traj_yn - y_mean)
        theta_cov = P - Sigma21 @ inv_Sigma11 @ Sigma21.transpose(-1, -2)

        # 预测实际轨迹
        y_pred = y_mean + phi @ theta_m
        y_cov = phi @ theta_cov @ phi.transpose(-1, -2)
        y_std = torch.sqrt(torch.diagonal(y_cov))

        # 记录预测结果
        self.log_pre = {"id": id, "flag_compare": flag_compare,
                    "t": data["t"][id], "x": traj_x.detach().cpu().numpy(),
                    "y": traj_y.detach().cpu().numpy().reshape(-1,), "yn": traj_yn.detach().cpu().numpy().reshape(-1,),
                   "y_pre": y_pred.detach().cpu().numpy().reshape(-1,), "y_std": y_std.detach().cpu().numpy().reshape(-1,),
                   "theta": theta_m.detach().cpu().numpy(), "theta_cov": theta_cov.detach().cpu().numpy()}

        return self.log_pre

    def traj_identify_predict_after_t0(self, id, t1=0, t2=4, data="train", flag_compare=False):
        """
        用 t0 以前的数据辨识 theta，再预测 t0 之后的 y

        模型:
            y = m(x) + phi(x) theta + w

        其中
            theta ~ N(0, P)
            w ~ N(0, sigma^2 I)

        参数
        ----
        id : int
            第 id 条轨迹
        t0 : float
            分界时刻。用 t < t0 的数据辨识 theta，用 t >= t0 的数据预测
        data : str
            "train" / "test"

        返回
        ----
        log_pre : dict
        """

        data_dict = self.dataset[data]
        device = self.args.device

        # ===== 读取整条轨迹 =====
        t = np.asarray(data_dict["t"][id])
        traj_x = Np2Tensor(data_dict["x"][id], device=device)
        traj_y = Np2Tensor(data_dict["y"][id], device=device)
        traj_yn = Np2Tensor(data_dict["yn"][id], device=device)

        # ===== 按 t0 划分 =====
        idx_hist = np.where((t1 < t) & (t < t2))[0]
        idx_pred = np.where((t <= t1) | (t >= t2))[0]

        if len(idx_hist) == 0:
            raise ValueError("t0 过小，t < t0 的历史数据为空，无法辨识 theta")
        if len(idx_pred) == 0:
            raise ValueError("t0 过大，t >= t0 的预测区间为空")

        # 历史段：用于辨识 theta
        x_hist = traj_x[idx_hist]
        yn_hist = traj_yn[idx_hist]

        # 未来段：用于预测 y
        x_pred = traj_x[idx_pred]
        t_pred = t[idx_pred]

        # ===== 预训练参数 =====
        if not flag_compare:
            P = self.ker.P0.to(device)
        else:
            P = torch.eye(self.ker.P0.shape[-1], device=device, dtype=self.ker.P0.dtype)

        # 历史段特征
        mean_hist = self.model.predict_mean(x_hist)
        phi_hist = self.model.predict_phi(x_hist)

        n_hist = x_hist.shape[-2]
        sigma2 = self.args.stdM * self.args.stdM
        Sigma_noise_hist = sigma2 * torch.eye(n_hist, dtype=phi_hist.dtype)
        Sigma_noise_hist = Sigma_noise_hist.to(device)

        # ===== 用历史数据做条件高斯推断 theta | y_hist =====
        # Sigma11 = phi_hist P phi_hist^T + sigma^2 I
        # Sigma21 = P phi_hist^T
        Sigma21 = P @ phi_hist.transpose(-1, -2)  # (dphi, n_hist)

        inv_Sigma11 = utils.woodbury_inverse_torch(
            Sigma_noise_hist, phi_hist, P
        )  # (n_hist, n_hist)

        resid_hist = yn_hist - mean_hist
        if resid_hist.ndim == 1:
            resid_hist = resid_hist.unsqueeze(-1)  # (n_hist,1)

        theta_m = Sigma21 @ inv_Sigma11 @ resid_hist

        # 使用伍德伯里公式逆运算整体计算协方差
        # P - P Φ^T (Φ P Φ^T + Σ_M^-1) Φ P = (P^-1 + Φ^T Σ_M^-1 Φ)^-1
        inv_Sigma_noise_hist = 1 / sigma2 * torch.eye(n_hist, dtype=phi_hist.dtype)
        inv_Sigma_noise_hist = inv_Sigma_noise_hist.to(device)
        inv_P = utils.inv_by_solve(P)

        theta_cov = utils.inv_by_solve(
            inv_P + phi_hist.transpose(-1, -2) @ inv_Sigma_noise_hist @ phi_hist
        )

        # 若输出想保持 1D
        theta_m_out = theta_m.squeeze(-1) if theta_m.shape[-1] == 1 else theta_m

        # ===== 用后验 theta 预测未来段 y =====
        mean_pred = self.model.predict_mean(x_pred)
        phi_pred = self.model.predict_phi(x_pred)

        if mean_pred.ndim == 1:
            mean_pred = mean_pred.unsqueeze(-1)

        y_pred = mean_pred + phi_pred @ theta_m  # (n_pred,1)

        y_cov = phi_pred @ theta_cov @ phi_pred.transpose(-1, -2)
        y_std = torch.sqrt(torch.diagonal(y_cov))

        P_cpu = P.cpu()
        theta_cov_cpu = theta_cov.cpu()
        phi_pred_cpu = phi_pred.cpu()
        phi_hist_cpu = phi_hist.cpu()
        y_cov_cpu = y_cov.cpu()


        # ===== 记录结果 =====
        self.log_pre = {
            "id": id,
            "t1": t1,
            "t2": t2,
            "flag_compare": flag_compare,

            "t": t,
            "x": traj_x.detach().cpu().numpy(),
            "y": traj_y.detach().cpu().numpy().reshape(-1, ),
            "yn": traj_yn.detach().cpu().numpy().reshape(-1, ),

            "t_pred": t_pred,
            "x_pred": x_pred.detach().cpu().numpy(),
            "y_pre": y_pred.detach().cpu().numpy().reshape(-1, ),
            "y_std": y_std.detach().cpu().numpy().reshape(-1, ),

            "theta": theta_m_out.detach().cpu().numpy(),
            "theta_cov": theta_cov.detach().cpu().numpy(),
        }

        return self.log_pre

    def plot_traj_identify_single(self, log_pre):

        id = log_pre["id"]
        flag = log_pre["flag_compare"]

        sys = self.dataset["sys"]
        sys.reset_W()
        y_nominal = sys.uncertainty(log_pre['x_pred'])[0]
        log_pre["y_nominal"] = y_nominal.reshape(-1,)

        fig = res.plot_traj_identify(log_pre)
        tag = "_compare" if flag else ""
        res.save_figure(
            fig, os.path.join(
                res.stage_output_dir(self.args.pics_path, "stage2"),
                f"traj_identify_{id}{tag}.png"))
        return fig

    def traj_epistemic_gmm(
            self,
            id,
            data="train",
            n_samples=100,
            include_noise=False,
    ):
        """
        对第 id 条轨迹的全部 x 应用模型，
        通过采样网络参数形成 GMM，计算认知不确定性。
        """

        data_dict = self.dataset[data]
        device = self.args.device

        self.model.eval()

        # ===== 读取整条轨迹 =====
        t = np.asarray(data_dict["t"][id])

        traj_x = Np2Tensor(data_dict["x"][id], device=device)
        traj_y = Np2Tensor(data_dict["y"][id], device=device)
        traj_yn = Np2Tensor(data_dict["yn"][id], device=device)

        sys = self.dataset["sys"]
        sys.reset_W()
        traj_x_np = Torch2Np(traj_x)
        y_nominal = sys.uncertainty(traj_x_np)[0]
        cov_th = res.cal_cov_theory(sys, self.args.std_a, traj_x_np)

        # ===== 只调用模型，不做在线辨识 =====
        with torch.no_grad():
            out = self.model.predict_gmm(
                traj_x,
                n_samples=n_samples,
                include_noise=include_noise,
                return_full_cov=True,
            )
            y_param_mean = self.model.predict_mean(traj_x)

        # ===== 去掉 batch 维 =====
        y_mean = out["mean"].squeeze(0)
        var_total = out["var_total"].squeeze(0)
        var_epi = out["var_epi"].squeeze(0)
        var_alea = out["var_alea"].squeeze(0)
        var_th = np.diagonal(cov_th).reshape(-1,)

        std_total = torch.sqrt(var_total.clamp_min(1e-12))
        std_epi = torch.sqrt(var_epi.clamp_min(1e-12))
        std_alea = torch.sqrt(var_alea.clamp_min(1e-12))
        std_th = np.sqrt(np.clip(np.diagonal(cov_th), 0.0, None)).reshape(-1,)

        cov_alea = out["cov_alea"].squeeze(0)
        cov_epi = out["cov_epi"].squeeze(0)
        cov_total = out["cov_total"].squeeze(0)

        # ===== 记录结果 =====
        self.log_epi_traj = {
            "id": id,
            "data": data,
            "n_samples": n_samples,
            "include_noise": include_noise,

            "t": t,
            "x": traj_x.detach().cpu().numpy(),
            "y": traj_y.detach().cpu().numpy().reshape(-1),
            "yn": traj_yn.detach().cpu().numpy().reshape(-1),
            "y_nominal": y_nominal.reshape(-1),
            "cov_th": cov_th,

            "y_gmm_mean": y_mean.detach().cpu().numpy().reshape(-1),
            "y_param_mean": y_param_mean.detach().cpu().numpy().reshape(-1),
            "component_mean": out["component_mean"].detach().cpu().numpy()[:, 0, :, 0],

            "std_total": std_total.detach().cpu().numpy().reshape(-1),
            "std_epi": std_epi.detach().cpu().numpy().reshape(-1),
            "std_alea": std_alea.detach().cpu().numpy().reshape(-1),
            "std_th": std_th,

            "var_total": var_total.detach().cpu().numpy().reshape(-1),
            "var_epi": var_epi.detach().cpu().numpy().reshape(-1),
            "var_alea": var_alea.detach().cpu().numpy().reshape(-1),
            "var_th": var_th,

            "cov_alea": cov_alea.detach().cpu().numpy(),
            "cov_epi": cov_epi.detach().cpu().numpy(),
            "cov_total": cov_total.detach().cpu().numpy(),

        }

        return self.log_epi_traj

    def grid_epistemic_gmm(
            self,
            x1_min=-4.0,
            x1_max=4.0,
            x2_min=-10.0,
            x2_max=10.0,
            n1=101,
            n2=101,
            n_samples=100,
            include_noise=False,
    ):
        """
        生成二维网格 x = [x1, x2]，
        调用学习到的模型参数分布，
        通过 GMM 计算网格上的预测均值和不确定性。

        返回结果中既保留 reshape(-1) 的一维形式，
        也保留 (n2, n1) 的二维网格形式，便于画热力图。

        网格排列：
            X1, X2 = np.meshgrid(x1, x2, indexing="xy")

        因此：
            X1.shape = (n2, n1)
            X2.shape = (n2, n1)
            x_grid.shape = (n1*n2, 2)
        """

        device = self.args.device

        self.model.eval()

        # ===== 生成二维网格 =====
        x1 = np.linspace(x1_min, x1_max, n1)
        x2 = np.linspace(x2_min, x2_max, n2)

        X1, X2 = np.meshgrid(x1, x2, indexing="xy")

        x_grid_np = np.stack(
            [X1.reshape(-1), X2.reshape(-1)],
            axis=-1
        )  # (n1*n2, 2)

        x_grid = Np2Tensor(x_grid_np, device=device)

        # 真实模型数据
        sys = self.dataset["sys"]
        sys.reset_W()
        y_nominal = sys.uncertainty(x_grid_np)[0]

        # A grid heatmap only needs pointwise variances.  Building the full
        # (n1*n2)^2 covariance matrix can consume several GB.
        var_th = res.cal_var_theory(sys, self.args.std_a, x_grid_np)

        # predict_gmm 要求输入为 (B, n, 2)
        x_in = x_grid.unsqueeze(0)  # (1, N, 2)

        # ===== GMM 预测 =====
        with torch.no_grad():
            out = self.model.predict_gmm(
                x_in,
                n_samples=n_samples,
                include_noise=include_noise,
                return_full_cov=False,
            )
            y_param_mean = self.model.predict_mean(x_in)

        y_mean = out["mean"].squeeze(0)
        var_total = out["var_total"].squeeze(0)
        var_epi = out["var_epi"].squeeze(0)
        var_alea = out["var_alea"].squeeze(0)

        std_total = torch.sqrt(var_total.clamp_min(1e-12))
        std_epi = torch.sqrt(var_epi.clamp_min(1e-12))
        std_alea = torch.sqrt(var_alea.clamp_min(1e-12))
        std_th = np.sqrt(var_th)

        # ===== 转 numpy，一维 =====
        y_gmm_mean = y_mean.detach().cpu().numpy().reshape(-1)

        std_total_np = std_total.detach().cpu().numpy().reshape(-1)
        std_epi_np = std_epi.detach().cpu().numpy().reshape(-1)
        std_alea_np = std_alea.detach().cpu().numpy().reshape(-1)

        var_total_np = var_total.detach().cpu().numpy().reshape(-1)
        var_epi_np = var_epi.detach().cpu().numpy().reshape(-1)
        var_alea_np = var_alea.detach().cpu().numpy().reshape(-1)

        # ===== 记录结果 =====
        self.log_grid_epi = {
            "x1": x1,
            "x2": x2,
            "X1": X1,
            "X2": X2,
            "x_grid": x_grid_np,
            "y_nominal": y_nominal.reshape(-1,),
            "n1": n1,
            "n2": n2,
            "n_samples": n_samples,
            "include_noise": include_noise,

            # 一维形式
            "y_gmm_mean": y_gmm_mean,
            "y_param_mean": y_param_mean.detach().cpu().numpy().reshape(-1),
            "std_total": std_total_np,
            "std_epi": std_epi_np,
            "std_alea": std_alea_np,
            "std_th": std_th,

            "var_total": var_total_np,
            "var_epi": var_epi_np,
            "var_alea": var_alea_np,
            "var_th": var_th,

            # 二维网格形式，方便 imshow / pcolormesh
            "Y_gmm_mean": y_gmm_mean.reshape(n2, n1),
            "STD_total": std_total_np.reshape(n2, n1),
            "STD_epi": std_epi_np.reshape(n2, n1),
            "STD_alea": std_alea_np.reshape(n2, n1),
            "STD_th": std_th.reshape(n2, n1),

            "VAR_total": var_total_np.reshape(n2, n1),
            "VAR_epi": var_epi_np.reshape(n2, n1),
            # Variance across sampled mean-network outputs only.  This does
            # not include phi/P0 function covariance or observation noise.
            "VAR_mean_output": var_epi_np.reshape(n2, n1),
            "VAR_alea": var_alea_np.reshape(n2, n1),
            "VAR_th": var_th.reshape(n2, n1),
        }

        return self.log_grid_epi

    def plot_learned_basis_functions(
            self, id=0, data="test", n_samples=20,
            x1_min=-4.0, x1_max=4.0, x2_min=-10.0, x2_max=10.0,
            n1=81, n2=81):
        """Plot posterior learned basis functions on a trajectory and 2-D plane."""
        data_dict = self.dataset[data]
        t = np.asarray(data_dict["t"][id]).reshape(-1)
        x_traj = Np2Tensor(data_dict["x"][id], device=self.args.device)

        with torch.no_grad():
            traj_out = self.model.predict_gmm(
                x_traj, n_samples=n_samples, include_noise=False,
                return_full_cov=False)

        traj_phi = traj_out["component_phi"].detach().cpu().numpy()
        actual_dz = int(traj_phi.shape[-1])
        configured_dz = int(getattr(self.args, "dz", actual_dz))
        if actual_dz != configured_dz:
            print(
                f"Warning: cfg.dz={configured_dz}, but the loaded model "
                f"contains {actual_dz} learned basis functions. Retrain the "
                f"model to apply the new dz setting.")
        basis_log = {
            "t": t,
            "n_basis": actual_dz,
            "P0": self.model.ker.P0.detach().cpu().numpy(),
            # (n_samples, n_time, n_basis): retain every BNN draw so the
            # plotting code can display actual posterior sample trajectories.
            "phi_traj_samples": traj_phi.squeeze(1),
            "phi_traj_mean": traj_phi.mean(axis=0).squeeze(0),
            "phi_traj_std": traj_phi.std(axis=0).squeeze(0),
        }
        figs = res.plot_learned_basis_functions(basis_log)
        res.save_figure(
            figs["basis_functions_trajectory"],
            os.path.join(
                res.stage_output_dir(self.args.pics_path, "stage2"),
                "bnn_basis_functions_trajectory.png"))
        return basis_log

    def plot_bnn_uncertainty_parameter_distribution(self):
        """Plot aggregate uncertainty distributions for mean and phi nets."""
        if not hasattr(self.model, "get_param_std"):
            raise AttributeError(
                "The loaded model does not expose variational parameter stds.")

        posterior_std = self.model.get_param_std()
        names = list(getattr(self.model, "mean_param_names", []))
        names += list(getattr(self.model, "phi_param_names", []))
        if len(names) != len(posterior_std):
            names = [f"parameter_{i}" for i in range(len(posterior_std))]

        split = int(getattr(self.model, "n_mean_param", 0))
        groups = []
        mean_values = []
        phi_values = []
        for i, (name, std) in enumerate(zip(names, posterior_std)):
            values = std.detach().cpu().numpy().reshape(-1)
            network = "mean" if i < split else "basis"
            if network == "mean":
                mean_values.append(values)
            else:
                phi_values.append(values)
            groups.append({
                "name": f"{network}: {name}",
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            })

        prior_values = getattr(self.model, "prior_std", [])
        prior_std = None
        if prior_values:
            prior_std = float(np.mean([
                p.detach().cpu().numpy().mean() for p in prior_values
            ]))

        uncertainty_log = {
            "groups": groups,
            "mean_values": (
                np.concatenate(mean_values) if mean_values
                else np.asarray([], dtype=float)
            ),
            "phi_values": (
                np.concatenate(phi_values) if phi_values
                else np.asarray([], dtype=float)
            ),
            "prior_std": prior_std,
        }
        fig = res.plot_bnn_uncertainty_parameter_distribution(
            uncertainty_log)
        res.save_figure(
            fig,
            os.path.join(
                res.stage_output_dir(self.args.pics_path, "stage2"),
                "bnn_uncertainty_parameter_distribution.png"))
        return uncertainty_log

    def plot_network_mean_parameter_distributions(self):
        """Plot parameter-centre distributions after Stage 1 and Stage 2."""
        train_log = utils.load_pickle(self.args.log_path)
        stage1 = train_log.get("stage1_network_mean_parameters")
        if stage1 is None:
            raise KeyError(
                "The training log has no Stage-1 network-parameter snapshot. "
                "Please train once with the current code before plotting it.")

        stage2 = {
            "mean_values": np.concatenate([
                parameter.detach().cpu().numpy().reshape(-1)
                for parameter in self.model.mean.parameters()
            ]),
            "phi_values": np.concatenate([
                parameter.detach().cpu().numpy().reshape(-1)
                for parameter in self.model.ker.phi.parameters()
            ]),
        }
        for stage_name, values in (("stage1", stage1), ("stage2", stage2)):
            fig = res.plot_bnn_mean_parameter_distribution(
                values, stage_label=stage_name.capitalize())
            res.save_figure(
                fig, os.path.join(
                    res.stage_output_dir(self.args.pics_path, stage_name),
                    "network_mean_parameter_distribution.png"))
        return {"stage1": stage1, "stage2": stage2}

    @torch.no_grad()
    def plot_network_parameter_mean_outputs(self, id=4, data="test"):
        """Plot Stage-1/2 parameter-centre mean and basis trajectory outputs."""
        train_log = utils.load_pickle(self.args.log_path)
        checkpoint = train_log.get("pretrained_mean_checkpoint")
        if not checkpoint or "phi_state_dict" not in checkpoint:
            raise KeyError(
                "A Stage-1 mean/phi checkpoint is required. Retrain with the "
                "current code before plotting the stage comparison.")

        data_dict = self.dataset[data]
        x_all = np.asarray(data_dict["x"])
        trajectory_id = min(max(int(id), 0), x_all.shape[0] - 1)
        x_np = x_all[trajectory_id]
        time_all = np.asarray(
            data_dict.get("t", np.arange(x_np.shape[0])[None, :]))
        time = time_all[trajectory_id] if time_all.ndim > 1 else time_all

        sys = self.dataset["sys"]
        sys.reset_W()
        nominal = np.asarray(sys.uncertainty(x_np)[0]).reshape(-1)

        stage1_model = copy.deepcopy(self.model)
        stage1_model.mean.load_state_dict(
            checkpoint["mean_state_dict"], strict=True)
        if "ker_state_dict" in checkpoint:
            stage1_model.ker.load_state_dict(
                checkpoint["ker_state_dict"], strict=True)
        else:
            stage1_model.ker.phi.load_state_dict(
                checkpoint["phi_state_dict"], strict=True)
        stage1_model.set_normalization(
            checkpoint["x_center"], checkpoint["x_scale"],
            checkpoint["y_center"], checkpoint["y_scale"])
        stage1_model.eval()

        figures = {}
        for stage_name, model in (
                ("stage1", stage1_model), ("stage2", self.model)):
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            x = torch.as_tensor(x_np, device=device, dtype=dtype)
            mean = model.predict_mean(x).detach().cpu().numpy().reshape(-1)
            phi = model.predict_phi(x).detach().cpu().numpy()
            n = min(np.asarray(time).size, nominal.size, mean.size, phi.shape[0])
            error = mean[:n] - nominal[:n]
            plot_log = {
                "stage_label": stage_name.capitalize(),
                "trajectory_id": trajectory_id,
                "time": np.asarray(time).reshape(-1)[:n],
                "nominal_mean": nominal[:n],
                "predicted_mean": mean[:n],
                "predicted_phi": phi[:n],
                "P0": model.ker.P0.detach().cpu().numpy(),
                "mse": float(np.mean(error ** 2)),
                "mae": float(np.mean(np.abs(error))),
            }
            figures[stage_name] = res.plot_network_parameter_mean_outputs(
                plot_log)
            res.save_figure(
                figures[stage_name], os.path.join(
                    res.stage_output_dir(self.args.pics_path, stage_name),
                    "network_parameter_mean_outputs.png"))
        return figures

    def plot_grid_epistemic_gmm(self, x1_min=-4.0, x1_max=4.0, x2_min=-10.0, x2_max=10.0,
            n1=101, n2=101, n_samples=100, include_noise=False):

        self.grid_epistemic_gmm(x1_min=x1_min, x1_max=x1_max,x2_min=x2_min,x2_max=x2_max,n1=n1,n2=n2, n_samples=n_samples,
            include_noise=include_noise)

        figs = res.plot_grid_epistemic_gmm(self.log_grid_epi)
        res.save_named_figures(
            figs, res.stage_output_dir(self.args.pics_path, "stage2"))
        return self.log_grid_epi

    def plot_traj_epistemic_gmm(self, id, data="train", n_samples=100, include_noise=False):

        self.traj_epistemic_gmm(id=id, data=data, n_samples=n_samples, include_noise=include_noise)

        figs = res.plot_traj_epistemic_gmm(
            self.log_epi_traj, include_2d=False)
        one_dimensional = {
            key: figs[key] for key in (
                "mean_curve", "mean_network_distribution", "variance_curve")
        }
        res.save_named_figures(
            one_dimensional,
            res.stage_output_dir(self.args.pics_path, "stage2"))





if __name__ == "__main__":


    print("Done...for now")

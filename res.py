from typing import Tuple, Optional, List, Union
from types import SimpleNamespace

from scipy.signal import ellip
from torch import Tensor
from numpy import ndarray

import numpy as np
import torch
import copy
import os
from pathlib import Path
from scipy.linalg import block_diag
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator
import utils
from gp_prior_learn.utils import Torch2Np


def save_figure(fig, path, dpi=300):
    """Centralized figure saving for the whole experiment pipeline."""
    path = str(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_named_figures(figures, output_dir, suffix=".png", dpi=300):
    """Save a {name: figure} mapping into one output directory."""
    saved = {}
    for name, fig in figures.items():
        path = os.path.join(str(output_dir), f"{name}{suffix}")
        saved[name] = save_figure(fig, path, dpi=dpi)
    return saved


def stage_output_dir(output_dir, stage):
    """Return the canonical output directory for one training stage."""
    if stage not in ("stage1", "stage2"):
        raise ValueError(f"Unknown training stage: {stage}")
    return os.path.join(str(output_dir), stage)


def show_all():
    """Display all figures using the backend selected by the entry point."""
    plt.show()


def plot_saved_training_loss(cfg, losses=None):
    """Load, draw and save current or legacy training-loss histories."""
    if losses is None:
        log = utils.load_pickle(cfg.log_path)
        phase_keys = ("losses_pretrain", "losses_mean", "losses_phi")
        phase_losses = {
            key: log.get(key) for key in phase_keys
            if log.get(key) is not None and np.asarray(log.get(key)).size
        }
        losses = phase_losses if phase_losses else log.get("losses")
        if losses is None or (
                not isinstance(losses, dict)
                and np.asarray(losses).size == 0):
            legacy_figures = {}
            for key, values in (
                    ("mean", log.get("losses_net_mean")),
                    ("phi", log.get("losses_net_std"))):
                if values is None or np.asarray(values).size == 0:
                    continue
                legacy_figures[key] = loss_show(
                    {f"losses_{key}": values},
                    name=f"losses_{key}",
                    title=f"Legacy {key} network training loss")
            if legacy_figures:
                save_named_figures(
                    legacy_figures,
                    stage_output_dir(cfg.pics_path, "stage2"),
                    suffix="_legacy.png")
                print(
                    "Loaded legacy loss fields from: "
                    f"{Path(cfg.log_path).resolve()}")
                return legacy_figures
            print(
                "No loss history exists in: "
                f"{Path(cfg.log_path).resolve()}")
            return None

    loss_log = losses if isinstance(losses, dict) else {"losses": losses}
    figures = {}
    stage1_losses = loss_log.get("losses_pretrain")
    if stage1_losses is not None and np.asarray(stage1_losses).size:
        figures["stage1"] = plot_training_loss_components({
            "losses_pretrain": stage1_losses})
        save_figure(
            figures["stage1"], os.path.join(
                stage_output_dir(cfg.pics_path, "stage1"),
                "training_loss.png"))

    stage2_log = {
        key: loss_log[key] for key in ("losses_mean", "losses_phi")
        if key in loss_log and np.asarray(loss_log[key]).size
    }
    if stage2_log:
        figures["stage2"] = plot_training_loss_components(stage2_log)
        save_figure(
            figures["stage2"], os.path.join(
                stage_output_dir(cfg.pics_path, "stage2"),
                "training_loss.png"))

    if figures:
        return figures
    fig = plot_training_loss_components(loss_log)
    save_figure(
        fig, os.path.join(
            stage_output_dir(cfg.pics_path, "stage2"),
            "training_loss.png"))
    return {"stage2": fig}


def plot_saved_mean_pretraining(cfg):
    """Restore the frozen mean from log.pkl and compare it with data.pkl."""
    train_log = utils.load_pickle(cfg.log_path)
    dataset = utils.load_pickle(cfg.dataset_path)
    diagnostics = train_log.get("mean_pretrain_diagnostics")
    if not diagnostics:
        print("No mean-pretraining diagnostics found.")
        return None
    checkpoint = train_log.get("pretrained_mean_checkpoint")
    if not checkpoint:
        print("No frozen mean-network checkpoint found in log.pkl.")
        return None
    fig = plot_mean_pretraining_diagnostics(
        diagnostics, train_log["model"], checkpoint, dataset)
    save_figure(
        fig, os.path.join(
            stage_output_dir(cfg.pics_path, "stage1"),
            "mean_pretraining_diagnostics.png"))

    trajectory_log = build_pretrained_trajectory_uncertainty_log(
        train_log["model"], checkpoint, dataset,
        std_a=cfg.std_a, trajectory_id=4)
    trajectory_fig = plot_pretrained_trajectory_uncertainty(trajectory_log)
    save_figure(
        trajectory_fig,
        os.path.join(
            stage_output_dir(cfg.pics_path, "stage1"),
            "pretrained_trajectory_uncertainty.png"))
    basis_fig = plot_network_parameter_mean_outputs(trajectory_log)
    save_figure(
        basis_fig,
        os.path.join(
            stage_output_dir(cfg.pics_path, "stage1"),
            "pretrained_basis_functions.png"))
    return {
        "diagnostics": fig,
        "trajectory_uncertainty": trajectory_fig,
        "basis_functions": basis_fig,
    }


@torch.no_grad()
def build_pretrained_trajectory_uncertainty_log(
        trained_model, checkpoint, dataset, std_a, trajectory_id=4):
    """Compute one trajectory using the frozen Stage-1 mean and kernel."""
    if "phi_state_dict" not in checkpoint:
        raise KeyError(
            "The Stage-1 checkpoint has no phi-network state. Retrain with "
            "the current code before plotting pretraining uncertainty.")

    model = copy.deepcopy(trained_model)
    model.mean.load_state_dict(checkpoint["mean_state_dict"], strict=True)
    if "ker_state_dict" in checkpoint:
        model.ker.load_state_dict(checkpoint["ker_state_dict"], strict=True)
    else:
        # Compatibility with the immediately preceding checkpoint format.
        # Its phi result is exact, but its P0 may already be the Stage-2 value.
        model.ker.phi.load_state_dict(
            checkpoint["phi_state_dict"], strict=True)
        print(
            "Warning: Stage-1 P0 is absent from the checkpoint; the "
            "pretraining uncertainty plot uses the P0 stored in the final model.")
    model.set_normalization(
        checkpoint["x_center"], checkpoint["x_scale"],
        checkpoint["y_center"], checkpoint["y_scale"])
    model.eval()

    test = dataset["test"]
    x_all = np.asarray(test["x"])
    if x_all.ndim < 3 or x_all.shape[0] == 0:
        raise ValueError("The test dataset contains no trajectory to plot.")
    trajectory_id = min(max(int(trajectory_id), 0), x_all.shape[0] - 1)
    x_np = x_all[trajectory_id]
    time_all = np.asarray(
        test.get("t", np.arange(x_np.shape[0])[None, :]))
    time = (time_all[trajectory_id] if time_all.ndim > 1 else time_all)
    time = np.asarray(time).reshape(-1)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x = torch.as_tensor(x_np, device=device, dtype=dtype)
    predicted_mean = model.predict_mean(x).detach().cpu().numpy().reshape(-1)
    predicted_phi = model.predict_phi(x).detach().cpu().numpy()
    predicted_cov = model.predict_cov(x).detach().cpu().numpy()
    predicted_cov = np.squeeze(predicted_cov)
    predicted_std = np.sqrt(
        np.clip(np.diagonal(predicted_cov), 0.0, None)).reshape(-1)

    sys = dataset["sys"]
    sys.reset_W()
    nominal_mean = np.asarray(sys.uncertainty(x_np)[0]).reshape(-1)
    theoretical_cov = cal_cov_theory(sys, std_a, x_np)
    theoretical_std = np.sqrt(
        np.clip(np.diagonal(theoretical_cov), 0.0, None)).reshape(-1)

    n = min(
        time.size, nominal_mean.size, theoretical_std.size,
        predicted_mean.size, predicted_std.size, predicted_phi.shape[0])
    error = predicted_mean[:n] - nominal_mean[:n]
    return {
        "trajectory_id": trajectory_id,
        "stage_label": "Stage 1",
        "time": time[:n],
        "nominal_mean": nominal_mean[:n],
        "theoretical_std": theoretical_std[:n],
        "predicted_mean": predicted_mean[:n],
        "predicted_phi": predicted_phi[:n],
        "P0": model.ker.P0.detach().cpu().numpy(),
        "predicted_std": predicted_std[:n],
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
    }


def plot_pretrained_trajectory_uncertainty(log):
    """Plot Stage-1 mean and uncertainty against the nominal trajectory."""
    time = np.asarray(log["time"]).reshape(-1)
    nominal = np.asarray(log["nominal_mean"]).reshape(-1)
    theoretical_std = np.asarray(log["theoretical_std"]).reshape(-1)
    prediction = np.asarray(log["predicted_mean"]).reshape(-1)
    predicted_std = np.asarray(log["predicted_std"]).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(
        time, nominal - theoretical_std, nominal + theoretical_std,
        color="tab:orange", alpha=.20, label="theoretical mean +/- 1 std")
    ax.plot(
        time, nominal, color="tab:orange", linestyle="--", linewidth=1.8,
        label="WingRock nominal mean")
    ax.fill_between(
        time, prediction - predicted_std, prediction + predicted_std,
        color="tab:blue", alpha=.20, label="pretrained mean +/- 1 std")
    ax.plot(
        time, prediction, color="tab:blue", linewidth=1.8,
        label="pretrained prediction")
    ax.text(
        .02, .98, f"MSE = {log['mse']:.4e}\nMAE = {log['mae']:.4e}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": .75})
    ax.set_title(
        f"Stage-1 Pretraining Result: Test Trajectory "
        f"{log['trajectory_id']}")
    ax.set_xlabel("time")
    ax.set_ylabel("Delta")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_network_parameter_mean_outputs(log):
    """Plot parameter-centre mean/basis outputs and the corresponding P0."""
    time = np.asarray(log["time"]).reshape(-1)
    nominal = np.asarray(log["nominal_mean"]).reshape(-1)
    predicted_mean = np.asarray(log["predicted_mean"]).reshape(-1)
    phi = np.asarray(log["predicted_phi"])
    if phi.ndim != 2:
        raise ValueError(f"Expected predicted_phi with 2 dimensions, got {phi.shape}")

    n_basis = phi.shape[-1]
    P0 = np.asarray(log["P0"])
    if P0.ndim != 2:
        raise ValueError(f"Expected P0 with 2 dimensions, got {P0.shape}")

    # One panel for the mean, one for each basis function, and one for P0.
    n_panels = n_basis + 2
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13, 3.6 * nrows), squeeze=False,
        sharex=True)

    ax = axes.flat[0]
    ax.plot(
        time, nominal, color="black", linestyle="--", linewidth=1.7,
        label="WingRock nominal mean")
    ax.plot(
        time, predicted_mean, color="tab:blue", linewidth=1.7,
        label="network parameter-mean prediction")
    ax.set_title(
        f"{log['stage_label']} mean output "
        f"(MSE={log['mse']:.3e}, MAE={log['mae']:.3e})")
    ax.set_ylabel("Delta")
    ax.legend()
    ax.grid(alpha=.3)

    for basis_id in range(n_basis):
        ax = axes.flat[basis_id + 1]
        ax.plot(
            time, phi[:, basis_id], color=f"C{basis_id % 10}",
            linewidth=1.6)
        ax.set_title(
            f"{log['stage_label']} parameter-mean basis phi_{basis_id}")
        ax.set_ylabel("basis output")
        ax.grid(alpha=.3)

    ax = axes.flat[n_basis + 1]
    ax.axis("off")
    cell_text = [[f"{value:.4g}" for value in row] for row in P0]
    row_labels = [f"{i}" for i in range(P0.shape[0])]
    col_labels = [f"{i}" for i in range(P0.shape[1])]
    table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        cellLoc="center", rowLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(max(6, 10 - max(P0.shape) // 3))
    table.scale(1.0, 1.25)
    ax.set_title(f"{log['stage_label']} P0")

    for ax in axes.flat[n_panels:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("time")
    fig.suptitle(
        f"{log['stage_label']} Network Parameter-Mean Outputs: "
        f"Test Trajectory {log['trajectory_id']}", fontsize=14)
    fig.tight_layout()
    return fig


def plot_mean_pretraining_diagnostics(log, trained_model, checkpoint, dataset):
    """Show robust loss views and predictions from the frozen pretrained mean."""
    losses = np.asarray(log.get("losses", []), dtype=np.float64)
    if losses.ndim > 1:
        # Train.optimize_step records [total, negative log likelihood, KL].
        # Stage 1 optimizes NLL directly, so plot that column explicitly.
        loss_column = 1 if losses.shape[1] > 1 else 0
        losses = losses[:, loss_column]

    model = copy.deepcopy(trained_model)
    model.mean.load_state_dict(checkpoint["mean_state_dict"])
    model.set_normalization(
        checkpoint["x_center"], checkpoint["x_scale"],
        checkpoint["y_center"], checkpoint["y_scale"])
    model.eval()
    device = next(model.parameters()).device

    test = dataset["test"]
    x_test_np = np.asarray(test["x"])
    y_test_np = np.asarray(test["yn"])
    x_test = torch.as_tensor(x_test_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_test = model.predict_mean(x_test).detach().cpu().numpy()

    # Recover the nominal WingRock model only when plotting.  No nominal
    # trajectory or diagnostic point cloud is stored in log.pkl.
    sys = dataset["sys"]
    sys.reset_W()
    nominal_test = np.asarray(
        sys.uncertainty(x_test_np.reshape(-1, x_test_np.shape[-1]))[0]
    ).reshape(*x_test_np.shape[:-1], 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    if losses.size:
        steps = np.arange(losses.size)
        ax.plot(steps, losses, color="tab:blue", alpha=.22, linewidth=.7,
                label="negative log likelihood")
        window = min(201, losses.size)
        if window > 1:
            smooth = np.convolve(
                losses, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], smooth, color="black", linewidth=1.5,
                    label=f"{window}-step moving average")
        ax.margins(y=.05)
    ax.set_title("Pretraining negative log likelihood — full history")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("negative log likelihood")
    ax.grid(alpha=.3)
    ax.legend()

    ax = axes[0, 1]
    if losses.size:
        start = losses.size // 2
        late_steps = np.arange(start, losses.size)
        ax.plot(late_steps, losses[start:], color="tab:blue", alpha=.3,
                linewidth=.7, label="negative log likelihood")
        window = min(201, losses.size - start)
        if window > 1:
            smooth = np.convolve(
                losses[start:], np.ones(window) / window, mode="valid")
            ax.plot(late_steps[window - 1:], smooth, color="black",
                    linewidth=1.5, label=f"{window}-step moving average")
        ax.margins(y=.05)
    ax.set_title("Pretraining negative log likelihood — late-stage detail")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("negative log likelihood")
    ax.grid(alpha=.3)
    ax.legend()

    ax = axes[1, 0]
    target = nominal_test.reshape(-1)
    prediction = pred_test.reshape(-1)
    max_points = min(20000, target.size)
    index = np.linspace(0, target.size - 1, max_points, dtype=int)
    ax.scatter(target[index], prediction[index], s=5, alpha=.18,
               color="tab:orange")
    lower = min(target[index].min(), prediction[index].min())
    upper = max(target[index].max(), prediction[index].max())
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    nominal_error = prediction - target
    mse = float(np.mean(nominal_error ** 2))
    mae = float(np.mean(np.abs(nominal_error)))
    ax.set_title(f"Frozen mean vs nominal test values\n"
                 f"nominal MSE={mse:.3g}, MAE={mae:.3g}")
    ax.set_xlabel("WingRock nominal mean")
    ax.set_ylabel("frozen pretrained mean")
    ax.grid(alpha=.3)

    ax = axes[1, 1]
    trajectory_id = min(4, x_test_np.shape[0] - 1)
    time = np.asarray(test.get("t", np.arange(x_test_np.shape[1])))[
        trajectory_id].reshape(-1)
    nominal = nominal_test[trajectory_id].reshape(-1)
    prediction = pred_test[trajectory_id].reshape(-1)
    n = min(time.size, nominal.size, prediction.size)
    ax.plot(time[:n], nominal[:n], color="tab:blue", linestyle="--",
            linewidth=1.7, label="WingRock nominal mean")
    ax.plot(time[:n], prediction[:n], color="tab:red", linewidth=1.5,
            label="frozen pretrained mean")
    ax.set_title(f"Frozen pretraining result: test trajectory {trajectory_id}")
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    ax.grid(alpha=.3)
    ax.legend()

    fig.suptitle("Mean-network pretraining diagnostics", fontsize=15)
    fig.tight_layout()
    return fig


# region dataset plot
def plot_trajectory_and_dynamics(log):
    t = log["t"]

    fig = plt.figure(figsize=(12, 10))

    # 1) 控制输入与实际舵偏
    plt.subplot(3, 1, 1)
    plt.plot(t[0, :], log["delta_cmd"][0, :], label="delta_cmd")
    plt.plot(t[0, :], log["v"][0, :], label="v (actual deflection)")
    plt.xlabel("Time (s)")
    plt.ylabel("Input")
    plt.title("Control command and actuator state")
    plt.grid(True)
    plt.legend()

    # 2) 状态轨迹
    plt.subplot(3, 1, 2)
    plt.plot(t[0, :], log["theta"][0, :], label="theta")
    plt.plot(t[0, :], log["p"][0, :], label="p")
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("State trajectories")
    plt.grid(True)
    plt.legend()

    # 3) 角加速度组成项
    plt.subplot(3, 1, 3)
    plt.plot(t[0, :], log["Delta"][0, :], label="Delta")
    plt.xlabel("Time (s)")
    plt.ylabel("Dynamics term")
    plt.title(r"Dynamics decomposition: $\dot p = L v + \Delta$")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    return fig

def plot_basis_functions(log):
    """Plot every available dataset basis without assuming a fixed dz."""
    t = np.asarray(log["t"])
    phi = np.asarray(log["Phi"])
    if phi.ndim == 2:
        phi = phi[None, ...]
    n_basis = phi.shape[-1]
    ncols = min(3, n_basis)
    nrows = int(np.ceil(n_basis / ncols))
    known_names = [
        "1", "theta", "p", "|theta| * p", "|p| * p", "theta^3"
    ]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows),
        squeeze=False, sharex=True)
    for j, ax in enumerate(axes.flat):
        if j >= n_basis:
            ax.axis("off")
            continue
        ax.plot(t[0], phi[0, :, j], linewidth=1.5)
        expression = known_names[j] if j < len(known_names) else f"basis {j}"
        ax.set_title(f"phi_{j} = {expression}")
        ax.set_xlabel("time")
        ax.set_ylabel(f"phi_{j}")
        ax.grid(alpha=.3)
    fig.tight_layout()
    return fig

def plot_dataset_io(log, use_noisy=False, title="Dataset"):
    """
    绘制整个数据集的输入输出关系

    参数
    ----
    log : dict
        包含 x, y, yn 的数据集
    use_noisy : bool
        是否使用 yn（带噪）
    """

    # ===== 取数据 =====
    x = log["x"]     # (N,n,2)
    y = log["yn"] if use_noisy else log["y"]
    t = log["t"]

    # ===== 转 numpy =====
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()

    # ===== reshape =====
    x_sc = x.reshape(-1, x.shape[-1])   # (N*n, 2)
    y_sc = y.reshape(-1, y.shape[-1])   # (N*n, 1)

    theta = x_sc[:, 0]
    p = x_sc[:, 1]
    Delta = y_sc[:, 0]
    N = t.shape[0]

    # ===== 作图 =====
    fig1 = plt.figure(figsize=(15, 4))

    # ===== 1. θ -> Δ =====
    plt.subplot(1, 3, 1)
    plt.scatter(theta, Delta, s=5, alpha=0.5)
    plt.xlabel("theta")
    plt.ylabel("Delta")
    plt.title(f"{title}: θ vs Δ")
    plt.grid(True)

    # ===== 2. p -> Δ =====
    plt.subplot(1, 3, 2)
    plt.scatter(p, Delta, s=5, alpha=0.5)
    plt.xlabel("p")
    plt.ylabel("Delta")
    plt.title(f"{title}: p vs Δ")
    plt.grid(True)

    # ===== 3. (θ,p) -> Δ =====
    plt.subplot(1, 3, 3)
    sc = plt.scatter(theta, p, c=Delta, s=5, cmap="viridis")
    plt.xlabel("theta")
    plt.ylabel("p")
    plt.title(f"{title}: Δ field")
    plt.grid(True)
    plt.colorbar(sc)

    plt.tight_layout()

    theta = log["theta"]
    p = log["p"]
    y = log["yn"] if use_noisy else log["y"]
    y = y.squeeze(-1)
    fig2 = plt.figure(figsize=(15, 4))

    # ===== 1. θ 随时间（不同轨迹）=====
    plt.subplot(1, 3, 1)
    for i in range(t.shape[0]):
        plt.plot(t[i], theta[i], alpha=0.3)
    plt.xlabel("time index")
    plt.ylabel("theta")
    plt.title("theta trajectories")
    plt.grid(True)

    # ===== 2. p 随时间（不同轨迹）=====
    plt.subplot(1, 3, 2)
    for i in range(t.shape[0]):
        plt.plot(t[i], p[i], alpha=0.3)
    plt.xlabel("time index")
    plt.ylabel("p")
    plt.title("p trajectories")
    plt.grid(True)

    # ===== 3. y(Delta) 随时间（不同轨迹）=====
    plt.subplot(1, 3, 3)
    for i in range(t.shape[0]):
        plt.plot(t[i], y[i], alpha=0.3)
    plt.xlabel("time index")
    plt.ylabel("Delta")
    plt.title("Delta trajectories")
    plt.grid(True)

    plt.tight_layout()
    return fig1, fig2

def plot_cov_theory(sys, std_a, x, title="Theoretical Covariance", cmap="viridis"):
    """
    计算并绘制给定数据上的理论协方差矩阵

    参数
    ----
    fun : callable
        基函数映射，输入 x，输出 phi
        要求输出形状为 (N, m) 或 (m, N)
    para_uncertain : array-like, shape (m, m)
        参数协方差矩阵
    x : array-like, shape (N,) 或 (N, d)
        输入数据
    title : str
        图标题
    cmap : str
        热力图颜色映射

    返回
    ----
    cov_th : ndarray, shape (N, N)
        理论协方差矩阵
    """

    cov_th = cal_cov_theory(sys, std_a, x)

    # ===== 画图 =====
    fig = plt.figure(figsize=(6, 5))

    plt.imshow(
        cov_th,
        origin="lower",
        aspect="auto",
        cmap=cmap
    )
    plt.xlabel("sample index")
    plt.ylabel("sample index")

    plt.title(title)
    plt.colorbar(label="Covariance")
    plt.tight_layout()

    return fig

def cal_cov_theory(sys, std_a, x):
    # ===== 转 numpy =====
    x = np.asarray(x)

    stds_a = [std_a, std_a]
    blocks = [v * v * np.ones((3, 3)) for v in stds_a]
    para_uncertain = block_diag(*blocks)

    # ===== 处理 x 维度 =====
    # 若 x 是一维数据点序列，转成 (N,1) 便于 fun 使用
    if x.ndim == 1:
        x_in = x[:, None]
    else:
        x_in = x

    # ===== 计算 phi =====
    sys.reset_W()
    phi = sys.phi(x_in)
    phi = np.asarray(phi) * sys.W.T

    # ===== 理论协方差 =====
    cov_th = phi @ para_uncertain @ phi.T  # (N,N)

    return cov_th

def cal_var_theory(sys, std_a, x):
    """Compute only diag(Phi Sigma Phi.T), avoiding an O(N^2) matrix."""
    x = np.asarray(x)
    x_in = x[:, None] if x.ndim == 1 else x
    stds_a = [std_a, std_a]
    para_uncertain = block_diag(
        *[v * v * np.ones((3, 3)) for v in stds_a]
    )
    sys.reset_W()
    phi = np.asarray(sys.phi(x_in)) * sys.W.T
    return np.einsum("ni,ij,nj->n", phi, para_uncertain, phi)

# endregion

# region train_res plot
@torch.no_grad()
def result_plot(args):

    log = utils.load_pickle(file_name=args.log_path)            # log记录训练结果和数据
    data = utils.load_pickle(file_name=args.dataset_path)       # data记录实验生成数据
    model = log["model"]
    sys = data["sys"]

    train_data = attach_prediction_to_log(model, data['train'])
    test_data = attach_prediction_to_log(model, data['test'])

    fig_train = plot_mean_var_compare(train_data, sys, args.std_a, title="train dataset")
    fig_test = plot_mean_var_compare(test_data, sys, args.std_a, title="test dataset")
    loss_figs = {}
    for key in ("losses", "losses_net_mean", "losses_net_std"):
        if key in log and np.asarray(log[key]).size:
            if key == "losses":
                loss_figs[key] = plot_training_loss_components(log, name=key)
            else:
                loss_figs[key] = loss_show(
                    log, name=key, title=key.replace("_", " ").title())
    stage2_dir = stage_output_dir(args.pics_path, "stage2")
    save_figure(fig_train, os.path.join(stage2_dir, "train_mean_uncertainty.png"))
    save_figure(fig_test, os.path.join(stage2_dir, "test_mean_uncertainty.png"))
    for key, fig in loss_figs.items():
        save_figure(fig, os.path.join(stage2_dir, f"{key}.png"))

    # Dense covariance heatmaps are intentionally disabled while the result
    # set is organized around one-dimensional trajectory diagnostics.

def plot_mean_var_compare(data, sys, std_a, title="Dataset", ci_scale=1.96):
    """
    从单个 log 中画：
    1. 学到的模型均值置信区间 ± ci_scale*std
    2. 真实值均值置信区间 ± ci_scale*std
    对比图

    要求 log 中至少包含：
        y          : 真实无噪值, shape (N,n,1) 或 (n,1)
        pred_mean  : 模型预测均值, shape (N,n,1) 或 (n,1)
        pred_std   : 模型预测标准差, shape (N,n,1) 或 (n,1)

    可选：
        t          : 时间, shape (N,n) 或 (n,)
    """
    required_keys = ["y", "pred_mean", "pred_std"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in log. Please save model predictions into log first.")

    y = np.asarray(data["y"])
    x = np.asarray(data["x"])[0]
    pred_mean = utils.ensure_3d(data["pred_mean"]) # (B, N, 1)
    pred_std = utils.ensure_3d(data["pred_std"])   # (B, N, 1)

    B, N, _ = y.shape

    if "t" in data:
        t = utils.ensure_2d(data["t"])             # (B, N)
        x_axis = t.mean(axis=0)
        x_label = "Time"
    else:
        x_axis = np.arange(N)
        x_label = "Sample index"

    # 对多条轨迹取统计均值和标准差
    sys.reset_W()
    y_mean = sys.uncertainty(x)[0].reshape(-1,)
    y_cov = cal_cov_theory(sys, std_a, x)
    y_std = np.sqrt(np.diag(y_cov))

    pred_mean = pred_mean.squeeze(0).squeeze(-1)
    pred_std = pred_std.squeeze(0).squeeze(-1)

    # 计算均值误差衡量
    diff = pred_mean - y_mean
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))

    fig = plt.figure(figsize=(10, 5))

    # 真实值均值和置信区间
    plt.plot(x_axis, y_mean, label="WingRock nominal mean")
    plt.fill_between(
        x_axis,
        y_mean - ci_scale * y_std,
        y_mean + ci_scale * y_std,
        alpha=0.25,
        label=f"WingRock nominal mean ± {ci_scale} std"
    )

    # 学习模型均值和置信区间
    plt.plot(x_axis, pred_mean, label="Predicted mean")
    plt.fill_between(
        x_axis,
        pred_mean - ci_scale * pred_std,
        pred_mean + ci_scale * pred_std,
        alpha=0.25,
        label=f"Predicted mean ± {ci_scale} std"
    )

    # 在图中显示 MSE / MAE
    metrics_text = f"MSE = {mse:.4e}\nMAE = {mae:.4e}"
    plt.text(
        0.02, 0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.xlabel(x_label)
    plt.ylabel("Delta")
    plt.title(f"{title}: mean and uncertainty comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig

@torch.no_grad()
def attach_prediction_to_log(model, data):
    """
    给 log 添加:
        pred_mean
        pred_std

    需要 log 里有 x
    取其中的第一组数据
    """
    if "x" not in data:
        raise KeyError("Missing key 'x' in log.")

    new_log = copy.deepcopy(data)

    x = data["x"]
    device = next(model.parameters()).device

    # ===== 转 tensor =====
    if not isinstance(x, torch.Tensor):
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    else:
        x_tensor = x.to(device=device, dtype=torch.float32)

    # ===== 处理维度 =====
    if x_tensor.ndim == 3:
        # (N, n, dx) → 取第一组
        x_tensor = x_tensor[0]

    elif x_tensor.ndim == 2:
        # (n, dx) → 保持
        pass

    elif x_tensor.ndim == 1:
        # (n,) → (n,1)
        x_tensor = x_tensor.unsqueeze(-1)

    else:
        raise ValueError(f"Unsupported shape: {tuple(x_tensor.shape)}")

    model.eval()


    K = model.predict_cov(x_tensor)
    pred_mean = model.predict_mean(x_tensor)

    if K.dim() == 2:
        pred_var = torch.diagonal(K, dim1=-2, dim2=-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported K shape: {K.shape}")

    pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-12))


    new_log["pred_mean"] = utils.Torch2Np(pred_mean)
    new_log["pred_std"] = utils.Torch2Np(pred_std)

    return new_log

def loss_show(log, name="losses_net_mean", title="Training Loss", save_path=None):
    """
    从 log 字典中读取 losses 并画图

    参数
    ----
    log : dict
        包含 'losses' 字段的日志字典
    title : str
        图标题
    save_path : str or None
        若不为 None，则保存图片到该路径
    show : bool
        是否显示图片

    返回
    ----
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """

    if name not in log:
        raise KeyError("log does not contain key 'losses'")

    losses = np.asarray(log[name], dtype=np.float64)
    # losses = np.log(losses)

    if losses.ndim != 1:
        losses = losses.reshape(-1)

    if losses.size == 0:
        raise ValueError(f"log[{name}] is empty")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(losses)), losses, label="train loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    return fig

@torch.no_grad()
def plot_cov_learn(model, sys, std_a, x, title="Kernel Covariance"):
    """
    绘制协方差矩阵 K 的热力图

    参数
    ----
    model : PriorLearn
    x     : (n, dx) 或 (N, n, dx)
    """
    # ===============计算预测协方差矩阵====================
    device = next(model.parameters()).device

    # ===== 转 tensor =====
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device=device, dtype=torch.float32)

    # ===== 维度处理 =====
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1,n,dx)

    elif x.ndim == 3:
        x = x[0:1]  # 只取一条轨迹 (1,n,dx)

    else:
        raise ValueError(f"Unsupported shape: {x.shape}")


    # ===== 调用 kernel =====
    K = model.predict_cov(x)

    # 兼容两种情况
    if isinstance(K, tuple):
        K = K[0]

    # ===== 转 numpy =====
    if K.ndim == 3:
        K = K[0]   # (n,n)

    K = K.detach().cpu().numpy()
    # ===== 计算理论协方差矩阵 =====
    x_np = Torch2Np(x)[0]
    y_cov = cal_cov_theory(sys, std_a, x_np)

    # ===== 作图 =====
    fig1 = plt.figure(figsize=(6, 5))

    plt.imshow(K, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="Covariance")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.title(title)
    plt.tight_layout()

    fig2 = plt.figure(figsize=(6, 5))

    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0)
    plt.imshow(y_cov - K, origin="lower", cmap="RdBu_r", aspect="auto", norm= norm)
    plt.colorbar(label="Covariance diff")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.title(title + " diff")
    plt.tight_layout()

    fig3 = plt.figure(figsize=(6, 5))

    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0)
    plt.imshow(y_cov, origin="lower", cmap="viridis", aspect="auto", norm=norm)
    plt.colorbar(label="Covariance theory")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.title(title + " theory")
    plt.tight_layout()

    return fig1, fig2, fig3

def plot_traj_identify(log_pre, title="Trajectory identification", sigma_scale=1.96):
    """
    展示轨迹预测结果

    参数
    ----
    log_pre : dict
        traj_identify 返回的字典，包含:
            x, y, yn, t, y_pre, y_std, theta, theta_cov
    title : str
        图标题
    sigma_scale : float
        标准差带倍数，默认画 ±2σ
    """

    t = np.asarray(log_pre["t"])
    y_nominal = np.asarray(log_pre["y_nominal"])
    y = np.asarray(log_pre["y"])
    yn = np.asarray(log_pre["yn"])

    t_pred = np.asarray(log_pre["t_pred"])
    t0 = t_pred.min()
    y_pre = np.asarray(log_pre["y_pre"])
    y_std = np.asarray(log_pre["y_std"])

    idx = np.searchsorted(t, t_pred)
    y_tr = y[idx]

    diff = y_pre - y_tr
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))

    fig = plt.figure(figsize=(8, 4.5))

    y_up = y_pre + sigma_scale * y_std
    y_low = y_pre - sigma_scale * y_std

    plt.plot(t_pred, y_nominal, color="blue", linestyle="--",
             linewidth=1.5, label="WingRock nominal mean")
    line = plt.plot(t_pred, y_pre, color="red", linewidth=2, label="predicted")

    plt.fill_between(
        t_pred,
        y_low,
        y_up,
        alpha=0.25,
        label=f"±{sigma_scale:.2f}σ",
        color=line[0].get_color(),
    )

    plt.axvline(t0, color="black", linestyle="--", linewidth=1)

    # 在图中显示 MSE / MAE
    metrics_text = f"MSE = {mse:.4e}\nMAE = {mae:.4e}"
    plt.text(
        0.02, 0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.xlabel("t")
    plt.ylabel(f"y")
    plt.title(f"{title}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return fig

# endregion

# region epistemic uncertainty analyze
def plot_uncertainty_diff_heatmap(
    model,
    sys,
    std_a,
    x1,
    x2=None,
    title="Theory - Learned Uncertainty",
):
    """
    绘制任意给定区域下：
        理论认知不确定性 - 学习认知不确定性
    的热力图

    参数
    ----
    model : PriorLearn
        含 model.ker 的模型
    sys : system
        用于 cal_cov_theory
    std_a : float
        理论协方差中的参数
    x1, x2 : tuple
        绘图区间
    """

    device = next(model.parameters()).device

    if x2 is None:
        x2 = x1
    n_grid1 = len(x1)
    n_grid2 = len(x2)
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    # =====================================================
    # 1. 构造二维网格
    # =====================================================

    X1, X2 = np.meshgrid(x1, x2)

    x_np = np.stack(
        [X1.reshape(-1), X2.reshape(-1)],
        axis=1
    )  # (n_grid1*n_grid, 2)

    x_torch = torch.as_tensor(
        x_np,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)  # (1, n, 2)

    # =====================================================
    # 2. 学习协方差
    # =====================================================
    model.eval()
    with torch.no_grad():
        K_learn = model.predict_cov(x_torch)

        if isinstance(K_learn, tuple):
            K_learn = K_learn[0]

        if K_learn.ndim == 3:
            K_learn = K_learn[0]  # (n, n)

        # 只取对角线作为每个状态点的不确定性
        cov_learn = torch.diagonal(K_learn).detach().cpu().numpy()

    cov_learn = cov_learn.reshape(n_grid1, n_grid2)

    # =====================================================
    # 3. 理论协方差
    # =====================================================
    K_theory = cal_cov_theory(sys, std_a, x_np)

    # 只取对角线
    cov_theory = np.diag(K_theory).reshape(n_grid1, n_grid2)

    # =====================================================
    # 4. 协方差预测误差
    # =====================================================
    cov_diff = np.abs(cov_theory - cov_learn)

    # =====================================================
    # 5. 均值差值
    # =====================================================
    sys.reset_W()
    mean_theory = sys.uncertainty(x_np)[0].reshape(-1, )
    mean_learn = model.predict_mean(x_torch).detach().cpu().numpy().reshape(-1, )
    mean_diff = ((mean_learn - mean_theory) ** 2).reshape(n_grid1, n_grid2)

    # =====================================================
    # 6. 作图
    # =====================================================
    fig = plt.figure(figsize=(6, 5))

    vmax = np.nanmax(np.abs(cov_diff))
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax
    )

    plt.imshow(
        mean_diff,
        origin="lower",
        extent=(x1[0], x1[-1], x2[0], x2[-1]),
        cmap="RdBu_r",
        aspect="auto",
        norm=norm
    )

    plt.colorbar(label="Theory - Learned Uncertainty")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(title)
    plt.tight_layout()

    return fig

def plot_epistemic_uncertainty(x1, x2, cov):
    """
    画两张图：
    1. cov 的完整协方差矩阵热力图 [N, N]
    2. diag(cov) 重排为 [len(x1), len(x2)] 后，在 x1-x2 平面上的方差热力图

    x1: [n1]
    x2: [n2]
    x_traj: [N, 2]
    cov: [N, N]
    """

    x1_np = utils.Torch2Np(x1)
    x2_np = utils.Torch2Np(x2)
    cov_np = utils.Torch2Np(cov).reshape(len(x1_np), len(x2_np))

    n1 = len(x1_np)
    n2 = len(x2_np)
    N = n1 * n2



    # =========================
    # 1. 完整协方差矩阵热力图
    # =========================
    fig1 = plt.figure(figsize=(8, 6))

    plt.pcolormesh(
        x1_np,
        x2_np,
        cov_np.T,
        shading="auto"
    )

    plt.colorbar(label="Epistemic Variance")

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Epistemic Uncertainty")

    plt.tight_layout()


    return fig1

def plot_grid_epistemic_gmm(
        log_grid,
        cmap="viridis",
):
    """
    绘制 grid_epistemic_gmm() 的结果

    返回
    -------
    figs : dict
    """

    X1 = log_grid["X1"]
    X2 = log_grid["X2"]

    Y_nominal = log_grid["y_nominal"].reshape(X1.shape)
    Y_gmm = log_grid["Y_gmm_mean"]

    STD_total = log_grid["STD_total"]
    STD_epi = log_grid["STD_epi"]
    STD_alea = log_grid["STD_alea"]
    STD_th = log_grid["STD_th"]

    VAR_th = log_grid["VAR_th"]
    VAR_total = log_grid["VAR_total"]
    VAR_epi = log_grid["VAR_epi"]
    VAR_alea = log_grid["VAR_alea"]
    VAR_mean_output = log_grid.get("VAR_mean_output", VAR_epi)

    Y_err = Y_gmm - Y_nominal

    figs = {}

    # ==================================================
    # nominal
    # ==================================================
    fig = plt.figure(figsize=(6, 5))

    plt.pcolormesh(
        X1,
        X2,
        Y_nominal,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(label="y")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Nominal Function")

    plt.tight_layout()

    figs["nominal"] = fig

    # ==================================================
    # gmm mean
    # ==================================================
    fig = plt.figure(figsize=(6, 5))

    plt.pcolormesh(
        X1,
        X2,
        Y_gmm,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(label="y")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("GMM Mean")

    plt.tight_layout()

    figs["gmm_mean"] = fig

    # ==================================================
    # variance of sampled mean-network outputs only
    # ==================================================
    fig = plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        X1, X2, VAR_mean_output,
        shading="auto", cmap="magma"
    )
    plt.colorbar(label="mean-network output variance")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Mean-network Output Variance")
    plt.tight_layout()
    figs["mean_output_variance"] = fig

    # ==================================================
    # mean error
    # ==================================================
    fig = plt.figure(figsize=(6, 5))

    vmax = np.max(np.abs(Y_err))

    plt.pcolormesh(
        X1,
        X2,
        Y_err,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    plt.colorbar(label="error")

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title("GMM Mean Error")

    plt.tight_layout()

    figs["mean_error"] = fig

    # ==================================================
    # total std
    # ==================================================
    fig = plt.figure(figsize=(6, 5))

    plt.pcolormesh(
        X1,
        X2,
        STD_total,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(label="std")

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title("Total Uncertainty")

    plt.tight_layout()

    figs["std_total"] = fig

    # ==================================================
    # epistemic std
    # ==================================================
    fig = plt.figure(figsize=(6, 5))

    plt.pcolormesh(
        X1,
        X2,
        STD_epi,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(label="std")

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title("Epistemic Uncertainty")

    plt.tight_layout()

    figs["std_epi"] = fig

    # ==================================================
    # aleatoric std
    # ==================================================
    fig = plt.figure(figsize=(6, 5))

    plt.pcolormesh(
        X1,
        X2,
        STD_alea,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(label="std")

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title("Aleatoric Uncertainty")

    plt.tight_layout()

    figs["std_alea"] = fig

    fig = plt.figure(figsize=(6, 5))

    plt.pcolormesh(
        X1,
        X2,
        VAR_th - VAR_alea,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(label="var")

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title("Theoretical Uncertainty Diff")

    plt.tight_layout()

    figs["var_diff"] = fig

    return figs

def plot_traj_epistemic_gmm(
        log_epi,
        cmap="viridis",
        cov_cmap="RdBu_r",
        include_2d=False,
):
    """
    绘制 traj_epistemic_gmm() 输出的 log_epi。

    包含：
    1. y, y_gmm_mean, y_nominal 随时间曲线
    2. 方差曲线 var_total / var_epi / var_alea
    3. 协方差热力图 cov_total / cov_epi / cov_alea，横纵轴为时间

    返回：
        figs: dict
    """

    t = np.asarray(log_epi["t"]).reshape(-1)

    y_nominal = np.asarray(log_epi["y_nominal"]).reshape(-1)
    y_gmm_mean = np.asarray(log_epi["y_gmm_mean"]).reshape(-1)
    y_param_mean = np.asarray(log_epi["y_param_mean"]).reshape(-1)
    component_mean = np.asarray(log_epi.get("component_mean", []))

    var_total = np.asarray(log_epi["var_total"]).reshape(-1)
    var_epi = np.asarray(log_epi["var_epi"]).reshape(-1)
    var_alea = np.asarray(log_epi["var_alea"]).reshape(-1)
    var_th = log_epi["var_th"].reshape(-1)
    std_th = np.asarray(
        log_epi.get("std_th", np.sqrt(np.clip(var_th, 0.0, None)))
    ).reshape(-1)

    cov_total = np.asarray(log_epi["cov_total"])
    cov_epi = np.asarray(log_epi["cov_epi"])
    cov_alea = np.asarray(log_epi["cov_alea"])
    cov_th = np.asarray(log_epi["cov_th"])

    # 若 cov 带 batch 维，例如 (1, n, n)，则去掉
    cov_total = np.squeeze(cov_total)
    cov_epi = np.squeeze(cov_epi)
    cov_alea = np.squeeze(cov_alea)

    figs = {}

    # ==================================================
    # 1. 均值曲线
    # ==================================================
    fig = plt.figure(figsize=(7, 4.5))

    plt.fill_between(
        t, y_nominal - std_th, y_nominal + std_th,
        color="tab:orange", alpha=.18, label="theoretical mean +/- 1 std")
    plt.plot(t, y_nominal, label="theoretical mean", linewidth=1.5,
             color="tab:orange", linestyle="--")
    if component_mean.ndim == 2:
        for sample in component_mean:
            plt.plot(t, sample, color="tab:blue", alpha=.12, linewidth=.8)
    std_total = np.sqrt(np.clip(var_total, 0.0, None))
    plt.fill_between(t, y_gmm_mean - std_total,
                     y_gmm_mean + std_total, color="tab:blue",
                     alpha=.18, label="BNN-GMM mean +/- 1 std")
    plt.plot(t, y_gmm_mean, label="BNN-GMM mean", linewidth=2,
             color="tab:blue")
    plt.plot(t, y_param_mean, label="mean network (parameter mean)",
             linewidth=1.8, color="tab:red", linestyle="-.")

    plt.xlabel("time")
    plt.ylabel("y")
    plt.title("Trajectory Mean Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    figs["mean_curve"] = fig

    # ==================================================
    # 2. distribution of mean-network predictions
    # ==================================================
    fig = plt.figure(figsize=(7, 4.5))
    if component_mean.ndim == 2 and component_mean.shape[0] > 0:
        lower = np.quantile(component_mean, .025, axis=0)
        upper = np.quantile(component_mean, .975, axis=0)
        sample_mean = np.mean(component_mean, axis=0)
        plt.fill_between(
            t, lower, upper, color="tab:blue", alpha=.22,
            label="sampled mean-network 95% interval")
        max_draws = min(20, component_mean.shape[0])
        draw_ids = np.linspace(
            0, component_mean.shape[0] - 1, max_draws, dtype=int)
        for sample in component_mean[draw_ids]:
            plt.plot(t, sample, color="tab:blue", alpha=.10, linewidth=.7)
        plt.plot(t, sample_mean, color="tab:blue", linewidth=2,
                 label="sample-average prediction")
    plt.plot(t, y_param_mean, color="tab:red", linestyle="-.",
             linewidth=1.8, label="parameter-mean prediction")
    plt.plot(t, y_nominal, color="black", linestyle="--", linewidth=1.5,
             label="nominal")
    plt.xlabel("time")
    plt.ylabel("mean-network output")
    plt.title("Mean-network Predictive Distribution on One Trajectory")
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    figs["mean_network_distribution"] = fig

    # ==================================================
    # 3. 方差曲线
    # ==================================================
    fig = plt.figure(figsize=(7, 4.5))

    plt.plot(t, var_total, label="total variance", linewidth=2)
    plt.plot(t, var_epi, label="epistemic variance", linewidth=2)
    plt.plot(t, var_alea, label="aleatoric variance", linewidth=2)
    plt.plot(t, var_th, label="theoretical covariance", linewidth=2)

    plt.xlabel("time")
    plt.ylabel("variance")
    plt.title("Trajectory Uncertainty Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    figs["variance_curve"] = fig

    if not include_2d:
        return figs

    # ==================================================
    # 5. 协方差热力图函数
    # ==================================================
    def _plot_cov(cov, title, key, label="covariance"):
        fig = plt.figure(figsize=(6, 5))

        vmax = np.nanmax(np.abs(cov))
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = None

        if vmax is None:
            plt.pcolormesh(
                t,
                t,
                cov,
                shading="auto",
                cmap=cmap
            )
        else:
            plt.pcolormesh(
                t,
                t,
                cov,
                shading="auto",
                cmap=cov_cmap,
                vmin=-vmax,
                vmax=vmax
            )

        plt.colorbar(label=label)
        plt.xlabel("time")
        plt.ylabel("time")
        plt.title(title)
        plt.tight_layout()

        figs[key] = fig

    _plot_cov(
        cov_total,
        title="Total Covariance",
        key="cov_total_heatmap",
    )

    _plot_cov(
        cov_epi,
        title="Epistemic Covariance",
        key="cov_epi_heatmap",
    )

    _plot_cov(
        cov_alea,
        title="Aleatoric Covariance",
        key="cov_alea_heatmap",
    )

    _plot_cov(
        cov_th - cov_alea,
        title="Theoretical Covariance Diff",
        key="cov_diff_heatmap",
    )

    return figs

def plot_learned_basis_functions(log_basis, cmap="coolwarm",
                                 max_sample_curves=20):
    """Visualize BNN posterior samples and summaries of learned bases."""
    t = np.asarray(log_basis["t"]).reshape(-1)
    phi_t = np.asarray(log_basis["phi_traj_mean"])
    phi_t_std = np.asarray(log_basis["phi_traj_std"])
    phi_t_samples = np.asarray(log_basis.get("phi_traj_samples", []))
    P0 = np.asarray(log_basis["P0"])
    n_basis = phi_t.shape[-1]
    n_panels = n_basis + 1
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    figs = {}

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             squeeze=False, sharex=True)
    for j, ax in enumerate(axes.flat):
        if j == n_basis:
            ax.axis("off")
            cell_text = [[f"{value:.4g}" for value in row] for row in P0]
            labels_y = [f"{i}" for i in range(P0.shape[0])]
            labels_x = [f"{i}" for i in range(P0.shape[1])]
            table = ax.table(
                cellText=cell_text, rowLabels=labels_y, colLabels=labels_x,
                cellLoc="center", rowLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(max(6, 10 - max(P0.shape) // 3))
            table.scale(1.0, 1.25)
            ax.set_title("Stage2 P0")
            continue
        if j > n_basis:
            ax.axis("off")
            continue
        if phi_t_samples.ndim == 3:
            n_draws = min(max_sample_curves, phi_t_samples.shape[0])
            draw_ids = np.linspace(
                0, phi_t_samples.shape[0] - 1, n_draws, dtype=int)
            for k, draw_id in enumerate(draw_ids):
                ax.plot(
                    t, phi_t_samples[draw_id, :, j],
                    color="tab:blue", alpha=0.18, linewidth=0.8,
                    label="BNN posterior samples" if k == 0 else None,
                )
        ax.fill_between(t, phi_t[:, j] - 2 * phi_t_std[:, j],
                        phi_t[:, j] + 2 * phi_t_std[:, j], alpha=.22,
                        color="tab:orange", label="sample mean ±2 std")
        ax.plot(t, phi_t[:, j], color="black", lw=2.0,
                label=f"sample mean phi_{j}")
        ax.set_title(f"Learned basis {j}: BNN sampled trajectories")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"phi_{j}(x(t))")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    figs["basis_functions_trajectory"] = fig

    if "phi_grid_mean" not in log_basis or "phi_grid_std" not in log_basis:
        return figs

    X1, X2 = log_basis["X1"], log_basis["X2"]
    phi_g = np.asarray(log_basis["phi_grid_mean"])
    phi_g_std = np.asarray(log_basis["phi_grid_std"])
    for values, suffix, label in (
            (phi_g, "mean", "posterior mean"),
            (phi_g_std, "std", "posterior std")):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)
        for j, ax in enumerate(axes.flat):
            if j >= n_basis:
                ax.axis("off")
                continue
            mesh = ax.pcolormesh(X1, X2, values[..., j], shading="auto",
                                 cmap=cmap if suffix == "mean" else "viridis")
            fig.colorbar(mesh, ax=ax, label=label)
            ax.set_title(f"Learned basis {j}: {label}")
            ax.set_xlabel("theta")
            ax.set_ylabel("p")
        fig.tight_layout()
        figs[f"basis_functions_grid_{suffix}"] = fig
    return figs

def plot_bnn_uncertainty_parameter_distribution(log_uncertainty,
                                                n_bins=30):
    """Histograms of posterior std for the complete mean and phi networks."""
    mean_std = np.asarray(
        log_uncertainty.get("mean_values", []), dtype=float).reshape(-1)
    phi_std = np.asarray(
        log_uncertainty.get("phi_values", []), dtype=float).reshape(-1)
    mean_std = mean_std[np.isfinite(mean_std) & (mean_std > 0)]
    phi_std = phi_std[np.isfinite(phi_std) & (phi_std > 0)]
    if mean_std.size == 0 or phi_std.size == 0:
        raise ValueError(
            "Both mean-network and phi-network uncertainty values are required.")

    all_std = np.concatenate([mean_std, phi_std])
    lower = max(float(np.min(all_std)), np.finfo(float).tiny)
    upper = float(np.max(all_std))
    if np.isclose(lower, upper):
        lower, upper = lower * 0.9, upper * 1.1
    # Use equal-width bins so the horizontal axis directly represents the
    # posterior standard-deviation magnitude on a linear scale.
    edges = np.linspace(lower, upper, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mean_counts, _ = np.histogram(mean_std, bins=edges)
    phi_counts, _ = np.histogram(phi_std, bins=edges)
    total_counts = mean_counts + phi_counts

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    axes[0].bar(
        centers, total_counts, width=widths, color="tab:purple",
        alpha=.82, edgecolor="white",
        label=f"all parameters (n={all_std.size})")
    axes[0].set_title("Combined mean + phi parameter uncertainty")
    axes[0].legend()

    axes[1].bar(
        centers, mean_counts, width=widths, color="tab:blue",
        alpha=.65, edgecolor="white",
        label=f"mean network (n={mean_std.size})")
    axes[1].bar(
        centers, phi_counts, width=widths, color="tab:orange",
        alpha=.55, edgecolor="white",
        label=f"phi network (n={phi_std.size})")
    axes[1].set_title("Uncertainty by network")
    axes[1].legend()

    prior_std = log_uncertainty.get("prior_std")
    for ax in axes:
        ax.set_xlabel("posterior uncertainty size (parameter std)")
        ax.set_ylabel("parameter count")
        ax.grid(axis="y", alpha=.3)
        if prior_std is not None and np.isfinite(prior_std) and prior_std > 0:
            ax.axvline(prior_std, color="black", linestyle="--",
                       linewidth=1.2, label="prior std")
        # Keep the view tightly restricted to bins containing posterior data.
        # In particular, an out-of-range prior marker must not create a large
        # empty horizontal region.
        margin = max((upper - lower) * 0.02, np.finfo(float).eps)
        ax.set_xlim(lower - margin, upper + margin)

    fig.tight_layout()
    return fig


def plot_bnn_mean_parameter_distribution(log_parameters, stage_label,
                                         n_bins=30):
    """Histograms of learned parameter centres for the mean and phi nets."""
    mean_values = np.asarray(
        log_parameters.get("mean_values", []), dtype=float).reshape(-1)
    phi_values = np.asarray(
        log_parameters.get("phi_values", []), dtype=float).reshape(-1)
    mean_values = mean_values[np.isfinite(mean_values)]
    phi_values = phi_values[np.isfinite(phi_values)]
    if mean_values.size == 0 or phi_values.size == 0:
        raise ValueError(
            "Both mean-network and phi-network parameter values are required.")

    all_values = np.concatenate([mean_values, phi_values])
    lower, upper = float(np.min(all_values)), float(np.max(all_values))
    if np.isclose(lower, upper):
        margin = max(abs(lower) * .1, .1)
        lower, upper = lower - margin, upper + margin
    edges = np.linspace(lower, upper, n_bins + 1)
    centers = .5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mean_counts, _ = np.histogram(mean_values, bins=edges)
    phi_counts, _ = np.histogram(phi_values, bins=edges)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    axes[0].bar(
        centers, mean_counts + phi_counts, width=widths,
        color="tab:purple", alpha=.82, edgecolor="white",
        label=f"all parameters (n={all_values.size})")
    axes[0].set_title(f"{stage_label}: combined mean + phi parameter centres")
    axes[0].legend()

    axes[1].bar(
        centers, mean_counts, width=widths, color="tab:blue",
        alpha=.65, edgecolor="white",
        label=f"mean network (n={mean_values.size})")
    axes[1].bar(
        centers, phi_counts, width=widths, color="tab:orange",
        alpha=.55, edgecolor="white",
        label=f"phi network (n={phi_values.size})")
    axes[1].set_title(f"{stage_label}: parameter centres by network")
    axes[1].legend()

    margin = max((upper - lower) * .02, np.finfo(float).eps)
    for ax in axes:
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2,
                   label="zero")
        ax.set_xlabel("learned parameter centre (weight / bias)")
        ax.set_ylabel("parameter count")
        ax.set_xlim(lower - margin, upper + margin)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    return fig

def plot_training_loss_components(log, name="losses"):
    """
    Plot the loss tuple recorded by Train.optimize_step:
        [total loss, negative log likelihood, weighted KL].

    Also accepts a one-dimensional legacy loss history.
    """
    labels = ["total loss", "negative log likelihood", "weighted KL"]
    colors = ["black", "tab:blue", "tab:orange"]

    phase_specs = [
        ("losses_pretrain", "Mean/basis deterministic pretraining"),
        ("losses_mean", "Sampled-BNN mean-network updates"),
        ("losses_phi", "Sampled-BNN phi-network updates"),
    ]
    available = [
        (key, title) for key, title in phase_specs
        if key in log and np.asarray(log[key]).size
    ]
    if not available:
        if name not in log:
            raise KeyError("log does not contain a recognized loss history")
        available = [(name, "Legacy joint training")]

    fig, axes = plt.subplots(
        len(available), 1, figsize=(11, 4 * len(available)),
        squeeze=False)
    for ax, (key, title) in zip(axes.flat, available):
        losses = np.asarray(log[key], dtype=np.float64)
        if losses.ndim == 1:
            losses = losses[:, None]
        elif losses.ndim > 2:
            losses = losses.reshape(losses.shape[0], -1)
        steps = np.arange(losses.shape[0])
        if key == "losses_pretrain":
            # Recorded tuple: [total loss, negative log likelihood, weighted KL].
            # Select NLL explicitly even though total == NLL in current Stage 1.
            loss_column = 1 if losses.shape[1] > 1 else 0
            ax.plot(
                steps, losses[:, loss_column], color="tab:blue",
                linewidth=1.25, label="negative log likelihood")
        else:
            n_components = min(losses.shape[1], len(labels))
            for j in range(n_components):
                ax.plot(steps, losses[:, j], color=colors[j], linewidth=1.25,
                        label=labels[j])
        if key == "losses_pretrain":
            # Let Matplotlib include the complete Stage-1 loss range. The
            # detailed diagnostics figure provides a separate late-stage view.
            ax.margins(y=.05)
        else:
            linthresh = 1.0
            ax.set_yscale(
                "symlog", linthresh=linthresh, linscale=1.0, base=10)
            ax.yaxis.set_major_locator(SymmetricalLogLocator(
                base=10, linthresh=linthresh, subs=(1.0,)))

            def format_symlog(value, _position):
                if np.isclose(value, 0.0):
                    return "0"
                sign = "−" if value < 0 else ""
                magnitude = abs(value)
                exponent = int(np.round(np.log10(magnitude)))
                if np.isclose(magnitude, 10.0 ** exponent):
                    return rf"${sign}10^{{{exponent}}}$"
                return f"{value:.2g}"

            ax.yaxis.set_major_formatter(FuncFormatter(format_symlog))
            ax.tick_params(axis="y", which="both", left=True,
                           labelleft=True, pad=5)
            ax.spines["left"].set_visible(True)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(
            "negative log likelihood" if key == "losses_pretrain"
            else "loss (symlog)")
        suffix = "negative log likelihood" if key == "losses_pretrain" else (
            "Monte Carlo ELBO")
        ax.set_title(title + f" ({suffix})")
        ax.grid(alpha=.3)
        ax.legend()

    fig.tight_layout()
    fig.subplots_adjust(left=0.13)
    return fig
# endregion











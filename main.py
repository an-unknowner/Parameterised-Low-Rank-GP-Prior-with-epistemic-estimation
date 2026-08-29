import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib
# Set the interactive backend exactly once, before importing pyplot anywhere.
matplotlib.use("TkAgg")

import numpy as np
import torch
from torch import Tensor

import utils, data, res
from learner import Train, Predict

PROJECT_DIR = Path(__file__).resolve().parent


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # 基本设置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument(
        "--dataset_path", type=str,
        default=str(PROJECT_DIR / "data" / "dataset.pkl"))
    parser.add_argument(
        "--log_path", type=str,
        default=str(PROJECT_DIR / "data" / "log.pkl"))
    parser.add_argument(
        "--pretrain_log_path", type=str,
        default=str(PROJECT_DIR / "data" / "pretrain_log.pkl"))
    parser.add_argument(
        "--pics_path", type=str,
        default=str(PROJECT_DIR / "res") + os.sep)

    # WingRock 仿真参数
    dt = 0.02
    parser.add_argument("--dt", type=float, default=dt)
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--flag_normalize", type=bool, default=True)

    # 训练激励：正弦控制指令 delta = A sin(2*pi*f*t + phase) + bias / 随机控制指令 amp幅度以内，限制变化量为max_step
    parser.add_argument("--dataset", type=str, default="rand", choices=["sin", "rand"])
    parser.add_argument("--train_time", type=float, default=15.0)
    parser.add_argument("--train_amp", type=float, default=5.0)
    parser.add_argument("--train_freq", type=float, default=0.50)
    parser.add_argument("--train_phase", type=float, default=0.0)
    parser.add_argument("--train_bias", type=float, default=0.0)
    parser.add_argument("--max_step", type=float, default=dt*5)
    parser.add_argument("--n_train_traj", type=int, default=500)
    parser.add_argument("--std_a", type=float, default=0.5)

    # 测试激励：略改相位/幅值，检查泛化
    parser.add_argument("--test_time", type=float, default=5.0)
    parser.add_argument("--test_amp", type=float, default=3.0)
    parser.add_argument("--test_freq", type=float, default=0.40)
    parser.add_argument("--test_phase", type=float, default=np.pi / 4)
    parser.add_argument("--test_bias", type=float, default=0.0)
    parser.add_argument("--n_test_traj", type=int, default=50)

    # PriorLearn 参数
    parser.add_argument("--model", type=str, default="exact")
    parser.add_argument("--ker_net", type=str, default="bfn", choices=["mlp_rbf", "tf", "bfn"])
    parser.add_argument("--dx", type=int, default=2)
    parser.add_argument("--dy", type=int, default=1)
    parser.add_argument("--dz", type=int, default=4)
    parser.add_argument('--stdM', type=Tensor, default=torch.tensor([1e-2]))
    parser.add_argument('--jitter', type=float, default=1e-10)
    parser.add_argument('--pretrain_only', type=bool, default=False,
                        help='if only conduct pretraining without BNN')

    # PriorLearn 神经网络先验参数
    parser.add_argument('--prior_mean', type=float, default=0.0,
                        help='deprecated: Stage-2 parameter prior is fixed at zero')
    parser.add_argument('--prior_std_rho', type=float, default=-1.5,
                        help='rho of the zero-mean parameter prior std')
    parser.add_argument('--init_std_rho', type=float, default=-2.5,
                        help='initial posterior parameter-std rho')
    parser.add_argument('--init_rho_noise', type=float, default=0.2)
    parser.add_argument('--jitter_total', type=float, default=1e-6)
    parser.add_argument('--mc_samples', type=int, default=30, help='number of ELBO MC integration samples')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of MC training')
    parser.add_argument('--beta_kl', type=float, default=1.0, help='KL coefficient for ELBO')
    parser.add_argument('--lambda_mean_guide', type=float, default=1.0,
                        help='weight of Stage-1 mean-output guidance in ELBO')
    parser.add_argument('--lambda_phi_guide', type=float, default=1.0,
                        help='weight of per-output Stage-1 phi guidance in ELBO')
    parser.add_argument(
        '--dropout', type=float, default=0.0,
        help='dropout probability for mean and basis MLP hidden layers')

    # 优化参数
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="1cycle", choices=["exp", "cos", "1cycle", "plateau"])
    parser.add_argument("--scheduler_pretrain", type=str, default="1cycle", choices=["exp", "cos", "1cycle"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument('--wd', type=float, default=0e-8, help='weight decay')
    parser.add_argument('--grad_max', type=float, default=1e-4, help='gradient max')
    parser.add_argument('--isSchdStep', type=bool, default=True, help='is scheduler step')
    parser.add_argument('--n_pre_mean', type=int, default=5000, help='number of alternating deterministic MLE pretraining cycles')
    parser.add_argument('--n_iters', type=int, default=5000, help='number of max iterations')
    parser.add_argument('--n_mean', type=int, default=5, help='number of mean steps')
    parser.add_argument('--n_cov', type=int, default=3, help='number of cov steps')

    cfg = parser.parse_args()
    return cfg

def train_model(cfg: argparse.Namespace):
    """Run training and save its log; do not create any figures here."""
    trainer = Train(cfg)
    trainer.run()
    return trainer


def plot_pretraining_results(cfg: argparse.Namespace):
    """Load the standalone Stage-1 log and regenerate all Stage-1 figures."""
    plot_cfg = SimpleNamespace(**vars(cfg))
    plot_cfg.log_path = cfg.pretrain_log_path
    return {
        "loss": res.plot_saved_training_loss(plot_cfg),
        "diagnostics": res.plot_saved_mean_pretraining(plot_cfg),
    }


def plot_full_training_results(cfg: argparse.Namespace, n_models=20):
    """Load the full-training log and generate Stage-1/2 result figures."""
    figures = {"loss": res.plot_saved_training_loss(cfg), "pretraining": res.plot_saved_mean_pretraining(cfg),
               "result": res.result_plot(cfg), "prediction": plot_prediction_results(cfg, n_models=n_models)}
    return figures

def dataset_generate(cfg: argparse.Namespace):
    data_generator = data.Data_Gen(cfg)
    data_generator.run()
    data_generator.data_show()
    return data_generator

def plot_prediction_results(cfg: argparse.Namespace, n_models=6):
    """Load the full model and generate prediction/uncertainty figures."""
    predictor = Predict(cfg)
    # log_pre = predictor.traj_identify_predict_after_t0(4, t1=0, t2=1, data="test", flag_compare=False)
    # predictor.plot_traj_identify_single(log_pre)
    # Two-dimensional grid and heatmap results are temporarily disabled.
    predictor.plot_traj_epistemic_gmm(id=4, data="test", n_samples=n_models)
    predictor.plot_learned_basis_functions(id=4, data="test", n_samples=n_models)
    predictor.plot_bnn_uncertainty_parameter_distribution()
    predictor.plot_network_mean_parameter_distributions()
    predictor.plot_network_parameter_mean_outputs(id=4, data="test")

    return predictor

if __name__ == "__main__":

    cfg = build_args()
    utils.set_seed(cfg.seed)
    n_models = 20

    # data_generator = dataset_generate(cfg)

    # train_model(cfg)
    if cfg.pretrain_only:
        plot_pretraining_results(cfg)
        message = "Pretraining finished."
    else:
        plot_full_training_results(cfg, n_models=n_models)
        message = "Full training finished."

    res.show_all()
    print(message)

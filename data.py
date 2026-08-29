import wingrock.wing_rock as wr
import argparse
from types import SimpleNamespace

from tqdm import tqdm
import numpy as np
import torch
import os

import utils, res
from wingrock.wing_rock import WingRock

def sin_gen(t: float, amp: float, freq: float, phase: float = 0.0, bias: float = 0.0) -> float:
    return bias + amp * np.sin(2.0 * np.pi * freq * t + phase)

class Data_Gen_sin:
    """
    用正弦控制指令激励 WingRock 系统，构造数据集，
    调用 prior_learn 中的 PriorLearn 模型学习不确定项 Delta(theta, p)，
    并在测试集上作图展示学习结果。

    关键关系：
        dot_p = L * v + Delta
        => Delta = dot_p - L * v
    因而可令：
        x = [theta, p]
        y = [Delta]
    """

    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.sys = WingRock(delta_t=cfg.dt, tau=cfg.tau)
        self.stdM = cfg.stdM

    def collect_data(self, duration: float, amp: float, freq: float, phase: float = 0.0, bias: float = 0.0,
                     std_a: float = 0.0) -> dict:
        """用正弦姿态指令采样系统轨迹，构造监督数据。"""
        self.sys = WingRock(delta_t=self.cfg.dt, tau=self.cfg.tau)      # 重置系统状态
        a = self.set_a(std_a=std_a)
        self.sys.set_W(a)                                # 设定采样轨迹随机参数
        steps = int(duration / self.sys.dt)

        t_log = []
        theta_log = []
        p_log = []
        v_log = []
        delta_cmd_log = []
        delta_true_log = []
        phi_true_log = []

        x_list = []
        y_list = []
        yn_list = []

        for _ in range(steps):
            t = self.sys.t
            theta_d = sin_gen(t=t, amp=amp, freq=freq, phase=phase, bias=bias)
            delta_cmd = self.sys.controller_update(theta_d=theta_d)
            theta, p, dot_p, Delta, Phi, v = self.sys.update(delta=delta_cmd)
            yn = Delta + np.random.randn()  * self.stdM.item()

            # 输入：x = [theta, p]
            x_list.append([theta, p])
            # 输出：y = [Delta]
            y_list.append([Delta])
            yn_list.append([yn])

            t_log.append(self.sys.t)
            theta_log.append(theta)
            p_log.append(p)
            v_log.append(v)
            delta_cmd_log.append(delta_cmd)
            delta_true_log.append(Delta)
            phi_true_log.append(Phi.ravel())

        logs = {
            "t": np.asarray(t_log, dtype=np.float32),           # (n, 1)
            "theta": np.asarray(theta_log, dtype=np.float32),   # (n, 1)
            "p": np.asarray(p_log, dtype=np.float32),           # (n, 1)
            "v": np.asarray(v_log, dtype=np.float32),           # (n, 1)
            "delta_cmd": np.asarray(delta_cmd_log, dtype=np.float32),   # (n, 1)
            "Delta": np.asarray(delta_true_log, dtype=np.float32),  # (n, 1)
            "Phi": np.asarray(phi_true_log, dtype=np.float32),  # (n, 6)
            "x": np.asarray(x_list, dtype=np.float32),          # (n, 2)
            "y": np.asarray(y_list, dtype=np.float32),          # (n, 1)
            "yn": np.asarray(yn_list, dtype=np.float32),        # (n, 1)
            "a": a.reshape(-1, 1),
        }

        return logs

    def set_a(self, std_a):
        a1, a2 = np.random.randn() * std_a + 1, np.random.randn() * std_a + 1
        a = np.array([[a1, a1, a1, a2, a2, a2]]).reshape(-1, 1)
        return a

    def run(self):
        train_list = []
        test_list = []

        # ===== 1. 采样训练集 =====
        for i in tqdm(range(self.cfg.n_train_traj), desc="Collecting train data"):
            logs = self.collect_data(
                duration=self.cfg.train_time,
                amp=self.cfg.train_amp,
                freq=self.cfg.train_freq,
                phase=self.cfg.train_phase,
                bias=self.cfg.train_bias,
                std_a=self.cfg.std_a,
            )
            train_list.append(logs)

        # ===== 2. 采样测试集 =====
        for i in tqdm(range(self.cfg.n_test_traj), desc="Collecting test data"):
            logs = self.collect_data(
                duration=self.cfg.test_time,
                amp=self.cfg.test_amp,
                freq=self.cfg.test_freq,
                phase=self.cfg.test_phase,
                bias=self.cfg.test_bias,
                std_a=self.cfg.std_a,
            )
            test_list.append(logs)

        # ===== 3. 堆叠函数 =====
        def stack_logs(log_list):
            keys = log_list[0].keys()
            stacked = {}

            for k in keys:

                stacked[k] = np.stack([log[k] for log in log_list], axis=0)

            return stacked

        train_logs = stack_logs(train_list)
        test_logs = stack_logs(test_list)

        # ===== 4. 保存 =====
        self.logs_tol = {
            "train": train_logs,
            "test": test_logs,
            "sys": self.sys,
        }

        print(f"Dataset saved to: {self.cfg.dataset_path}")
        print(f"Train shape x: {train_logs['x'].shape}")
        print(f"Test shape x: {test_logs['x'].shape}")
        utils.save_pickle(self.logs_tol, self.cfg.dataset_path)

    def data_show(self):

        if hasattr(self, "logs_tol"):
            train_data = self.logs_tol['train']
            test_data = self.logs_tol['test']
        else:
            raise RuntimeError("No simulation data found. Please run() first, then call the plotting function.")

        fig_train_bf = res.plot_basis_functions(train_data)
        fig_train_dy = res.plot_trajectory_and_dynamics(train_data)
        fig_test_bf = res.plot_basis_functions(test_data)
        fig_test_dy = res.plot_trajectory_and_dynamics(test_data)
        fig_train_total_sc, fig_train_total_traj = res.plot_dataset_io(train_data)
        fig_cov_th = self.cov_plot(train_data)


        output = os.path.join(self.cfg.pics_path, "dataset")
        res.save_named_figures({
            "train_bf": fig_train_bf,
            "train_dy": fig_train_dy,
            "test_bf": fig_test_bf,
            "test_dy": fig_test_dy,
            "train_total_sc": fig_train_total_sc,
            "train_total_traj": fig_train_total_traj,
            "cov_th": fig_cov_th,
        }, output)

        print("Dataset pics drawn")

    def cov_plot(self, data):
        """取数据集中的第一条曲线绘制协方差图像"""
        x = data["x"][0, :, :]
        fig = res.plot_cov_theory(self.sys, self.cfg.std_a, x)

        return fig

class Data_Gen_rand(Data_Gen_sin):
    """
    继承 Data_Gen_sin，将 theta_d 的生成方式由正弦信号改为随机受限信号。

    约束：
        1) |theta_d[k]| <= amp
        2) |theta_d[k] - theta_d[k-1]| <= max_step

    可选：
        用平滑系数 alpha 控制随机序列的平滑程度
    """

    def __init__(self, cfg: argparse.Namespace):
        super().__init__(cfg)

    def generate_theta_d_sequence(
        self,
        steps: int,
        amp: float,
        max_step: float,
        theta0: float = 0.0,
        alpha: float = 0.5,
    ):
        """
        生成受限随机参考信号 theta_d 序列

        参数
        ----
        steps : int
            序列长度
        amp : float
            theta_d 总幅值限制，满足 |theta_d| <= amp
        max_step : float
            相邻两步最大变化量
        theta0 : float
            初值
        alpha : float
            一阶平滑系数，越接近1越平滑
        rand_std : float
            原始随机扰动强度

        返回
        ----
        theta_d_seq : ndarray, shape (steps,)
        """

        theta_d_seq = np.zeros(steps, dtype=np.float32)
        theta_d_seq[0] = np.clip(theta0, -amp, amp)

        for k in range(1, steps):
            # 先生成平滑随机驱动
            w = alpha * theta_d_seq[k - 1] + (1.0 - alpha) * np.random.uniform(-amp, amp)

            # 对本步增量限幅
            # delta = np.clip(w, -max_step, max_step)

            # 计算候选值
            # theta_next = theta_d_seq[k - 1] + delta

            # 对总幅值限幅
            # theta_next = np.clip(theta_next, -amp, amp)
            theta_next = np.clip(w, -amp, amp)

            theta_d_seq[k] = theta_next

        return theta_d_seq

    def collect_data(
        self,
        duration: float,
        amp: float,
        std_a: float = 0.0,
        max_step: float = 0.1,
        alpha: float = 0.5,
        theta0: float = 0.0,
    ) -> dict:
        """
        用受限随机姿态指令采样系统轨迹，构造监督数据。

        amp:
            theta_d 的绝对幅值上限
        max_step:
            theta_d 相邻时刻最大变化量
            若未给出，则默认取 amp 的 5%
        """
        self.sys = WingRock(delta_t=self.cfg.dt, tau=self.cfg.tau)
        a = self.set_a(std_a=std_a)
        self.sys.set_W(a)

        steps = int(duration / self.sys.dt)

        if max_step is None:
            max_step = 0.05 * amp

        # 一次性生成整段随机参考信号
        theta_d_seq = self.generate_theta_d_sequence(
            steps=steps,
            amp=amp,
            max_step=max_step,
            theta0=theta0,
            alpha=alpha,
        )

        t_log = []
        theta_d_log = []
        theta_log = []
        p_log = []
        v_log = []
        delta_cmd_log = []
        delta_true_log = []
        phi_true_log = []

        x_list = []
        y_list = []
        yn_list = []

        for k in range(steps):
            theta_d = float(theta_d_seq[k])

            delta_cmd = self.sys.controller_update(theta_d=theta_d)
            theta, p, dot_p, Delta, Phi, v = self.sys.update(delta=delta_cmd)
            yn = Delta + np.random.randn() * self.stdM.item()

            x_list.append([theta, p])
            y_list.append([Delta])
            yn_list.append([yn])

            t_log.append(self.sys.t)
            theta_d_log.append(theta_d)
            theta_log.append(theta)
            p_log.append(p)
            v_log.append(v)
            delta_cmd_log.append(delta_cmd)
            delta_true_log.append(Delta)
            phi_true_log.append(Phi.ravel())

        logs = {
            "t": np.asarray(t_log, dtype=np.float32),
            "theta_d": np.asarray(theta_d_log, dtype=np.float32),
            "theta": np.asarray(theta_log, dtype=np.float32),
            "p": np.asarray(p_log, dtype=np.float32),
            "v": np.asarray(v_log, dtype=np.float32),
            "delta_cmd": np.asarray(delta_cmd_log, dtype=np.float32),
            "Delta": np.asarray(delta_true_log, dtype=np.float32),
            "Phi": np.asarray(phi_true_log, dtype=np.float32),
            "x": np.asarray(x_list, dtype=np.float32),
            "y": np.asarray(y_list, dtype=np.float32),
            "yn": np.asarray(yn_list, dtype=np.float32),
            "a": a.reshape(-1, 1),
        }

        return logs

    def run(self):
        train_list = []
        test_list = []

        # ===== 1. 采样训练集 =====
        for i in tqdm(range(self.cfg.n_train_traj), desc="Collecting train data"):
            logs = self.collect_data(
                duration=self.cfg.train_time,
                amp=self.cfg.train_amp,
                std_a=self.cfg.std_a,
                max_step=self.cfg.max_step,
                alpha=getattr(self.cfg, "train_alpha", 0.5),
                theta0=getattr(self.cfg, "train_theta0", 0.0),
            )
            train_list.append(logs)

        # ===== 2. 采样测试集 =====
        for i in tqdm(range(self.cfg.n_test_traj), desc="Collecting test data"):
            logs = self.collect_data(
                duration=self.cfg.test_time,
                amp=self.cfg.test_amp,
                std_a=self.cfg.std_a,
                max_step=getattr(self.cfg, "test_max_step", 0.1),
                alpha=getattr(self.cfg, "test_alpha", 0.5),
                theta0=getattr(self.cfg, "test_theta0", 0.0),
            )
            test_list.append(logs)

        # ===== 3. 堆叠函数 =====
        def stack_logs(log_list):
            keys = log_list[0].keys()
            stacked = {}

            for k in keys:
                stacked[k] = np.stack([log[k] for log in log_list], axis=0)

            return stacked

        train_logs = stack_logs(train_list)
        test_logs = stack_logs(test_list)

        # ===== 4. 保存 =====
        self.logs_tol = {
            "train": train_logs,
            "test": test_logs,
            "sys": self.sys,
        }

        print(f"Dataset saved to: {self.cfg.dataset_path}")
        print(f"Train shape x: {train_logs['x'].shape}")
        print(f"Test shape x: {test_logs['x'].shape}")

        utils.save_pickle(self.logs_tol, self.cfg.dataset_path)


def Data_Gen(cfg: argparse.Namespace):
    dataset_map = {
        'sin': Data_Gen_sin,
        'rand': Data_Gen_rand,
    }
    if cfg.dataset not in dataset_map:
        raise ValueError(f"Unknown model name: {cfg.dataset}")
    return dataset_map[cfg.dataset](cfg)




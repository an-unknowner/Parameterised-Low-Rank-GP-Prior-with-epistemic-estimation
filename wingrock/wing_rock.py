# -*- coding:utf-8 -*-
# @author  : Zheng Tengjie
# @time    : 2023/12/23 17:02
# @function: the dynamical system of wing rock.
# @version : V1

import numpy as np
from wingrock.method import *
from wingrock.utils_wr import inv_softplus, ToTensor
import torch.nn as nn

class WingRock():
    """Dynamical system of wing rock"""
    def __init__(self, delta_t=0.05, tau=0.1):
        """
        delta_t : time_step [float]
        """

        self.dt = delta_t
        self.t = 0.0 #初始时间
        self.theta = 0.0 # 初始滚转角
        self.p = 0.0 # 初始滚转角速度
        self.L = 3 # control effective 控制量系数，与舵偏相乘得到角速度的一部分

        #新增一阶惯性环节舵偏延迟
        self.tau = tau  # 惯性环节时间常数 T
        self.v = 0.0  # [新增状态] 实际舵偏角 (Actuator deflection)

        # parameters of uncertainty ？
        self.W_learn = nn.Parameter(inv_softplus(torch.zeros(6)), requires_grad=True)
        self.W_init = np.array([0.8, 0.2314, 0.6918, -0.6245, 0.0095, 0.0214]).reshape(-1, 1) # 原始参数
        self.W = self.W_init

        # 延迟处理
        self.delta_re = []

        # ctrl params
        self.theta_f = 0.
        self.p_f = 0.
        self.v_f = 0.0
        self.T_theta = 0.5
        self.T_p = 0.3
        self.T_v = 0.3
        self.K1 = 2
        self.K2 = self.K1 * 3
        self.K3 = self.K2 * 3


    def set_W(self, a):
        # 输入拉偏倍数和偏置
        self.W = np.array(self.W_init * a).reshape(-1, 1)

    def set_L(self, L):
        self.L = np.array(L).ravel()[0]

    def reset_W(self):
        self.W = self.W_init

    def controller_update(self, theta_d):
        theta_d = np.array(theta_d).ravel()[0]

        # 角度环
        theta_f_old = self.theta_f
        self.theta_f += (theta_d - self.theta_f) / self.T_theta * self.dt
        dot_theta_f = (self.theta_f - theta_f_old) / self.dt

        p_d = self.K1 * (self.theta_f - self.theta) + dot_theta_f

        # 角速度环
        p_f_old = self.p_f
        self.p_f += (p_d - self.p_f) / self.T_p * self.dt
        dot_p_f = (self.p_f - p_f_old) / self.dt

        dot_p_d = self.K2 * (self.p_f - self.p) + dot_p_f
        x = np.array([self.theta, self.p])
        Delta, _ = self.uncertainty(x)
        Delta = Delta.item()
        v_d = (dot_p_d - Delta) / self.L

        # 舵环
        v_f_old = self.v_f
        self.v_f += (v_d - self.v_f) / self.T_v * self.dt
        dot_v_f = (self.v_f - v_f_old) / self.dt

        dot_v_d = self.K3 * (self.v_f - self.v) + dot_v_f
        delta = (dot_v_d + self.v / self.tau) * self.tau

        return delta

    def update(self, delta):
        """update the system by control input delta
        arguments:
        delta : control input [float]
        returns:
        theta : roll angle [float, deg]
        p : roll rate [float, deg/s]
        dot_p : roll acceleration [float, deg/s^2]
        Delta: uncertainty [float, deg/s^2]
        """

        # 保存当前的控制指令，供 _dot_X 计算 dv/dt 使用
        self.delta_cmd = np.array(delta).item()

        # [关键修改] 状态向量扩展为 3 维: [theta, p, v]
        current_state = np.array([self.theta, self.p, self.v])

        # RKM_4 积分，向前推演一步
        X_next = RKM_4(current_state, self._dot_X, self.dt)

        # 更新状态
        self.theta, self.p, self.v = X_next[0], X_next[1], X_next[2]
        self.t += self.dt

        x = np.array([self.theta, self.p])
        self.Delta, self.Phi = self.uncertainty(x)
        self.Delta = self.Delta.item()

        # [关键修改] 计算角加速度时，使用实际舵偏角 self.v (即 v)，而不是指令 self.delta_cmd
        self.dot_p = self.L * self.v + self.Delta

        # 返回值增加了 self.v 方便绘图观察延迟
        return self.theta, self.p, self.dot_p, self.Delta, self.Phi, self.v

    def uncertainty(self, x):
        """calculate the uncertainty
        arguments:
        x: np.array([theta, p])
        theta : roll angle [float, deg]
        p : roll rate [float, deg/s]
        returns:
        Delta : uncertainty [ndarray, deg/s^2]
        """

        Phi = self.phi(x)
        Delta = Phi @ self.W # 两个向量内积得多项式

        return Delta, Phi

    def phi(self, x):
        x = np.asarray(x)

        n = x.shape[-1]
        if n != 2:
            # 在所有维度中找到长度为2的轴
            axes = [i for i, s in enumerate(x.shape) if s == 2]
            if not axes:
                raise ValueError("No dimension of size 2 found in x.shape")

            # 把该轴移动到最后一维
            x = np.moveaxis(x, axes[0], -1)

        theta = x[..., 0]
        p = x[..., 1]

        phi = np.vstack([
            np.ones_like(theta),
            theta,
            p,
            np.abs(theta) * p,
            np.abs(p) * p,
            theta ** 3
        ])

        return phi.T

    def _dot_X(self, X):
        """ODE of the system"""
        # X.ravel() 会把 X 按行主序（C-order）“展平成一维数组”（返回一个一维视图，不拷贝数据）
        theta = X.ravel()[0]
        p = X.ravel()[1]
        v = X.ravel()[2]  # [新增] 当前的实际舵偏角

        x = np.array([theta, p])
        Delta,_ = self.uncertainty(x)
        Delta = Delta.item()

        # 1. d(theta)/dt = p
        dot_theta = p

        # 2. d(p)/dt = Delta + L * v  (注意这里是 v，不是 delta_cmd)
        dot_p = self.L * v + Delta

        # 3. d(v)/dt = (u - v) / T   (一节惯性环节微分方程)
        # self.delta_cmd 是控制指令 u
        dot_v = (self.delta_cmd - v) / self.tau

        dot_X = np.array([dot_theta, dot_p, dot_v])

        return dot_X


class Test():
    def __init__(self):
        np.random.seed(1)
        torch.manual_seed(1)

        # 初始化 WingRock，设定时间常数 tau (例如 0.2秒)
        self.obj = WingRock(delta_t=0.02, tau=0.3)

        # PID 控制器 (参数可能需要根据延迟情况微调，延迟会导致系统变差)
        self.pid = PID(delta_t=self.obj.dt, Kp=4.0, Ki=2.0, Kd=2.0, flag_filter=True, filter_order=2, omega=1.7)

        self.sigma_noise = 0.2

        # [修改] 使用 list 初始化存储容器
        self.log_t = []
        self.log_theta = []
        self.log_theta_d = []
        self.log_p = []
        self.log_delta = []  # 控制指令
        self.log_v = []  # 实际舵偏 (观察延迟用)
        self.log_Delta = []

    def run(self):
        tend = 50  # 仿真时长 (秒)
        steps = int(tend / self.obj.dt)

        print(f"Simulation started. Total steps: {steps}, Tau: {self.obj.tau}s")

        for ii in range(steps):
            # 1. 生成期望信号
            # self.theta_d = np.sign(np.sin(2 * np.pi / 15 * self.obj.t)) * 1.5
            # self.theta_d += np.sign(np.sin(2 * np.pi / 15 * self.obj.t + np.pi / 2)) * 1.5
            if (ii % 20) == 0:
                self.theta_d = np.random.uniform(-6.0, 6.0)

            # 2. PID 计算控制指令 delta
            # self.delta = self.pid.update(self.theta_d, self.obj.theta)
            self.delta = self.obj.controller_update(self.theta_d)

            # 3. 系统更新 (物理步进)
            self.obj.update(delta=self.delta)

            # 4. 观测噪声 (用于可能的滤波反馈，这里仅用于记录)
            self.y = self.obj.theta + np.random.randn() * self.sigma_noise

            # 5. [修改] 将数据存入 list
            self.log_t.append(self.obj.t)
            self.log_theta.append(self.obj.theta)
            self.log_theta_d.append(self.theta_d)
            self.log_p.append(self.obj.p)
            self.log_delta.append(self.delta)  # 记录指令
            self.log_v.append(self.obj.v)  # 记录实际值
            self.log_Delta.append(self.obj.Delta)

            if ii % 100 == 0:
                print(f'Step {ii}/{steps}, Time: {self.obj.t:.2f}s')

        print('Simulation finished!')
        self.plot_results()

    def plot_results(self):
        """画图函数"""
        # 转为 numpy array 方便绘图
        t = np.array(self.log_t)
        theta = np.array(self.log_theta)
        theta_d = np.array(self.log_theta_d)
        delta = np.array(self.log_delta)
        v = np.array(self.log_v)

        plt.figure(figsize=(12, 8))

        # 子图1: 跟踪效果
        plt.subplot(2, 1, 1)
        plt.plot(t, theta_d, 'r--', label='Reference (theta_d)', linewidth=1.5)
        plt.plot(t, theta, 'b-', label='Response (theta)', linewidth=1.5)
        plt.title('WingRock Roll Angle Tracking')
        plt.ylabel('Angle (deg)')
        plt.grid(True)
        plt.legend()

        # 子图2: 舵机延迟效果 (重点观察这里)
        plt.subplot(2, 1, 2)
        plt.plot(t, delta, 'g--', label='Command (delta)', linewidth=1.5, alpha=0.7)
        plt.plot(t, v, 'k-', label='Actual Deflection (v)', linewidth=1.5)
        plt.title(f'Actuator Dynamics (Lag Tau={self.obj.tau}s)')
        plt.ylabel('Control Surface (deg)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()

        # 可以在这里保存图片
        # plt.savefig('sim_result.png')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    test = Test()
    test.run()
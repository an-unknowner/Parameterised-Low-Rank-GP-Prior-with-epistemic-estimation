from functools import wraps
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch


def first_order_lowpass_filter_zero_phase(pwm, sampling_rate=1000, cutoff_freq=50):
    """
    对PWM信号的4个通道分别应用双向一阶惯性低通滤波器（模拟零相位特性）
    """
    # 参数校验
    if pwm.ndim != 2 or pwm.shape[1] != 4:
        raise ValueError("输入pwm的形状必须为 (N, 4)")

    if cutoff_freq >= sampling_rate / 2:
        raise ValueError("截止频率必须小于奈奎斯特频率（采样率的一半）")

    # 计算滤波系数
    dt = 1.0 / sampling_rate
    alpha = 2 * np.pi * cutoff_freq * dt
    alpha = alpha / (1 + alpha)

    # 初始化输出数组
    filtered_pwm = np.zeros_like(pwm)

    # 对每个通道分别处理
    for channel in range(4):
        # 正向滤波
        forward_filtered = np.zeros_like(pwm[:, channel])
        forward_filtered[0] = pwm[0, channel]

        for i in range(1, len(pwm)):
            forward_filtered[i] = (1 - alpha) * forward_filtered[i - 1] + alpha * pwm[i, channel]

        # 反向滤波（从末尾向开头）
        backward_filtered = np.zeros_like(pwm[:, channel])
        backward_filtered[-1] = forward_filtered[-1]

        for i in range(len(pwm) - 2, -1, -1):
            backward_filtered[i] = (1 - alpha) * backward_filtered[i + 1] + alpha * forward_filtered[i]

        # 取反向滤波结果作为最终输出
        filtered_pwm[:, channel] = backward_filtered

    return filtered_pwm

def first_order_lowpass_filter(pwm, sampling_rate=1000, cutoff_freq=50):
    """
    对PWM信号的4个通道分别应用一阶惯性低通滤波器

    参数:
    pwm : numpy.ndarray
        输入信号，形状为 (N, 4) 的ndarray，N是数据长度，4是通道数
    sampling_rate : float, 可选
        采样频率（Hz），默认为1000Hz
    cutoff_freq : float, 可选
        截止频率（Hz），默认为50Hz

    返回:
    numpy.ndarray
        滤波后的信号，形状与输入相同 (N, 4)
    """
    # 参数校验
    if pwm.ndim != 2 or pwm.shape[1] != 4:
        raise ValueError("输入pwm的形状必须为 (N, 4)")

    if cutoff_freq >= sampling_rate / 2:
        raise ValueError("截止频率必须小于奈奎斯特频率（采样率的一半）")

    # 计算滤波系数
    dt = 1.0 / sampling_rate  # 采样时间间隔
    alpha = 2 * np.pi * cutoff_freq * dt  # 滤波系数计算
    alpha = alpha / (1 + alpha)  # 归一化系数

    # 初始化输出数组
    filtered_pwm = np.zeros_like(pwm)

    # 对每个通道分别应用一阶惯性滤波
    for channel in range(4):
        # 初始化第一个值
        filtered_pwm[0, channel] = pwm[0, channel]

        # 递归应用一阶惯性滤波
        for i in range(1, len(pwm)):
            filtered_pwm[i, channel] = (1 - alpha) * filtered_pwm[i - 1, channel] + alpha * pwm[i, channel]

    return filtered_pwm

class DataRecorder():
    def __init__(self, capality=None, flag_save=False, filepath = r'./'):
        self.flag_save = flag_save  # 若为真，则会在database第一次满了和以后每更新一轮的时候进行保存
        self.filepath = filepath    # 数据保存的路径

        self.database = {}          # 数据库
        self.empty_flag = True
        self.full_flag = False
        self.number_data = 0        # 接收到的数据总数，不等于数据库中的数据数量
        self.number_update = 0      # database更新完一轮的次数
        self.capality = capality    # 数据库的容量

    def data_add(self, data_name, data_vector):
        if self.empty_flag == True:
            for name, data in zip(data_name, data_vector):
                self.database[name] = np.array(data).reshape((1, -1))
            self.empty_flag = False
        else:
            if self.full_flag == False:
                for name, data in zip(data_name, data_vector):
                    self.database[name] = np.vstack(( self.database[name], np.array(data).reshape((1, -1)) ))
            else:
                for name, data in zip(data_name, data_vector):
                    self.database[name] = np.vstack(( self.database[name], np.array(data).reshape((1, -1)) ))
                    self.database[name] = self.database[name][1:, :]  # 先进先出

        self.number_data += 1

        flag_updata_finish = False  # database是否更新完一轮
        if not self.capality == None:
            if self.number_data >= self.capality:
                self.full_flag = True

            if np.abs(self.number_data % self.capality) < 1e-2 and self.number_data > 0:
                flag_updata_finish = True
                self.number_update += 1

            if flag_updata_finish and self.flag_save:
                # 保存当前的database
                filename = self.filepath + str(self.number_update) + '.txt'
                self.dictionary_save(self.database, filename)

        return flag_updata_finish

    def dictionary_save(self, dict, filename):
        # 字典变量保存
        keys_list = list(dict.keys())
        file = open(filename, 'w')  # 写入会覆盖

        for key in keys_list:
            s = key + ' '
            file.write(s)

            data = dict[key].ravel().tolist()
            s = str(data).replace('[', '').replace(']', '')  # 去除[],这两行代码按数据不同，可以选择
            s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
            file.write(s)

        file.close()
        print("dictionary save successfully")

    def dictionary_read(self, filename):
        # 字典变量读取
        file = open(filename, 'r')

        dict = {}
        arr = []
        while 1:
            s = file.readline()
            if s == '':
                break
            s = s.splitlines()[0]  # 去掉末尾的换行符
            s = s.split(' ')  # 按空格分割
            index = 0
            for ss in s:
                if index == 0:
                    key = ss
                else:
                    if ss != '':
                        arr.append(float(ss))
                index += 1
            dict[key] = np.array(arr)
            arr = []

        file.close()

        return dict

def Np2Num(x):
    return np.array(x).ravel()[0]

def Np2Torch(x, device=None):
    if isinstance(x, torch.Tensor):
        y = x.to(torch.float32)
    else:
        y = torch.tensor(x, dtype=torch.float32)

    if device is not None:
        return y.to(device)
    return y

def Torch2Np(x):
    return x.detach().to('cpu').numpy()

def NpCol(x):
    return np.array(x).reshape((-1, 1))

def NpRow(x):
    return np.array(x).reshape((1, -1))

def scatter_3D_plot(ax, x, y, z, s=0.1, color=None, xlabel=None, ylabel=None, zlabel=None):
    ax.scatter3D(x, y, z, s=s, color=color)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    if zlabel != None:
        ax.set_zlabel(zlabel)

def error_bar_plot(t, x, e, label='e', alpha=0.2, color='yellow', ax=None):
    # 误差带绘制
    x = x.ravel()
    e = e.ravel()
    if t is None:
        t = np.arange(x.size).ravel()
    else:
        t = t.ravel()

    if ax is None:
        plt.fill_between(t, x - e, x + e, alpha=alpha, color=color, label=label)
    else:
        ax.fill_between(t, x - e, x + e, alpha=alpha, color=color, label=label)

class Filter_linear():
    """
    线性滤波器
    """
    def __init__(self, delta_t, order=2, omega=2.0, y0=None):
        # delta_t: 积分步长
        # order: 滤波器阶数
        # omega: 滤波器的极点的相反数
        self.dt = delta_t
        self.order = order
        self.omega = omega
        self.set_system()
        if y0 is None:
            self.y = np.zeros(( self.order, 1 ))
        else:
            self.y = np.zeros(( self.order, 1 ))
            self.y[0, 0] = np.array(y0).ravel()[0]

    def set_system(self):
        self.B = np.zeros(( self.order, 1 ))
        self.C = np.zeros(( 1, self.order ))
        self.C[0, 0] = 1.0
        self.A = np.eye(self.order - 1)
        self.A = np.hstack(( np.zeros((self.order-1, 1)), self.A ))
        a_vec = np.zeros(( 1, self.order ))
        for ii in range(self.order):
            a_vec[0, ii] = -self.omega**(self.order-ii) * com_num(self.order, ii)
        self.A = np.vstack(( self.A, a_vec ))
        self.B[-1, 0] = -a_vec[0, 0]

    def update(self, x, dt=None):
        if dt is None:
            dt = self.dt
        x = np.array(x).reshape((-1, 1))
        dot_y = self.A @ self.y + self.B @ x
        self.y = self.y + dot_y * dt

        return Np2Num(self.C @ self.y)

def meshgrid_eval(X1, X2, f):
    """Meshgrid evaluation
    Arguments:
        X1 : input meshgrid matrix 1 [ndarray: dim_input1 x dim_input2]
        X2 : input meshgrid matrix 2 [ndarray: dim_input1 x dim_input2]
        f : evaluation function [function: ndarray -> ndarray]
    Returns
        output : evaluation result [ndarray: dim_output x dim_input1 x dim_input2]
    """

    for ii in range(X1.shape[0]):
        for jj in range(X1.shape[1]):
            in_single = np.array([X1[ii, jj], X2[ii, jj]])
            out_single = f(in_single).reshape(-1, 1)
            if ii == 0 and jj == 0:
                output = np.zeros((out_single.shape[0], X1.shape[0], X1.shape[1]))
            output[:, ii, jj] = out_single.ravel()

    return output

def meshgrid_cal(input, f, num_output):
    # 功能：进行meshgrad矩阵的函数计算
    # 输入：
    # input 一个列表，每个元素是输入一个维度的数据
    # f 函数句柄
    # num_output f输出的个数
    # 输出：
    # output 若num_output为1则是一个矩阵，若为=大于1则是一个列表，列表包含num_output个矩阵

    shape_data = input[0].shape
    if num_output == 1:
        output = np.zeros_like(input[0]).ravel()
    elif num_output > 1:
        output0 = np.zeros_like(input[0]).ravel()
        output = []
        for ii in range(num_output):
            output.append(np.copy(output0))

    for ii in range(input[0].size):
        input_array = np.zeros((len(input), 1)).ravel()
        for jj in range(len(input)):
            input_array[jj] = input[jj].ravel()[ii]
        if num_output == 1:
            output[ii] = f(input_array)
        elif num_output > 1:
            Out = f(input_array)
            for kk in range(num_output):
                output[kk][ii] = Out[kk]

    if num_output == 1:
        output = output.reshape(shape_data)
    elif num_output > 1:
        for kk in range(num_output):
            output[kk] = output[kk].reshape(shape_data)

    return output

def surface_plot(ax, X, Y, Z, alpha=1, label='', cmap=None, color=None):
    # 曲面图绘制
    surf = ax.plot_surface(X, Y, Z, alpha=alpha, label=label, cmap=cmap, color=color)
    # 解决图例报错
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    # plt.legend()

    return surf

def save_pickle(data, file_name, flag_show=True):
    # 将data保存成pkl文件
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()
    if flag_show:
        print(file_name + ' save successfully')

def load_pickle(file_name, flag_show=True):
    # pkl文件读取
    f = open(file_name, 'rb')
    data = pickle.load(f)
    f.close()
    if flag_show:
        print(file_name + ' load successfully')

    return data

def Factorial(n):
    # n的阶乘
    i = 1
    F = 1
    while i <= n:
        F *= i
        i += 1

    return F

def per_num(a, b):
    # 计算排列数A_a^b
    return Factorial(a) / Factorial(a - b)

def com_num(a, b):
    # 计算组合数C_a^b
    return Factorial(a) / Factorial(a - b) / Factorial(b)

def timing(func):
    @wraps(func)  # 保留func的属性
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print('%r took %f s\n' % (func.__name__, stop - start))
        sys.stdout.flush()  # 立即刷新输出
        return res

    return wrapper

def RKM_4(y, f, h):
    if h == 0:
        return y

    K0 = y
    K1 = f(y)
    K2 = f(y + 0.5 * h * K1)
    K3 = f(y + 0.5 * h * K2)
    K4 = f(y + h * K3)
    Y = K0 + h / 6 * (K1 + K2 * 2 + K3 * 2 + K4)
    # Y = y+f(y) * h

    return Y

def save_show(flag_save, flag_show, filename, fig, dpi=None, flag_tight=True):
    if flag_save:
        if flag_tight:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(filename, dpi=dpi)
    if not flag_show:
        plt.close(fig)

def set_legend(ax, rate_margin=0.15, ncol=5, fontsize=8, loc='upper right'):
    """Set the format of legend
    Arguments:
        rate_margin : rate of the top margin [float]
        ncol : number of the columns of the legend [int]
        fontsize : font size of the legend [int]
        loc : location of the legend [str]
    """
    Ylim = ax.set_ylim()
    d = Ylim[1] - Ylim[0]
    ax.set_ylim(Ylim[0], Ylim[1] + rate_margin * d)
    ax.legend(loc=loc, fontsize=fontsize, ncol=ncol)


class PID():
    def __init__(self, delta_t, Kp=10.0, Ki=2.0, Kd=0.1, flag_filter=False, filter_order=2, omega=1.0, xd_0=None):
        self.delta_t =delta_t
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.flag_filter = flag_filter
        if self.flag_filter:
            self.filter = Filter_linear(delta_t=delta_t, order=filter_order, omega=omega, y0=xd_0)

        self.e_old = 0
        self.ei = 0  # 跟踪误差的积分

    def update(self, xd, x):
        xd = np.array(xd).ravel()[0]
        x = np.array(x).ravel()[0]
        if self.flag_filter:
            self.xd_filtered = Np2Num(self.filter.update(xd))
        else:
            self.xd_filtered = xd
        e = self.xd_filtered - x
        dot_e = (e - self.e_old) / self.delta_t
        u = self.Kp * e + self.Ki * self.ei + self.Kd * dot_e  # PID控制
        self.ei += e * self.delta_t
        self.e_old = e

        return u


class Test():
    def __init__(self):
        pass

    def run(self):
        pass

    def result_plot(self):
        pass

if __name__ == '__main__':
    pass

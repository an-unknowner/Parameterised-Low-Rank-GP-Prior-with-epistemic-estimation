import pickle
import random
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import ndarray
from torch import Tensor


class ITrain():
    def __init__(self, logpath='.log/'):
        self.logpath = logpath
        self.loss_best = np.inf

    def update_log_best(self, loss, log):
        if loss < self.loss_best:
            self.loss_best = loss
            self.save_log(log, name='log_best.pkl')

    def save_log(self, log, name='log.pkl'):
        save_pickle(log, self.logpath + name, flag_show=False)

    def load_log(self, name='log.pkl'):
        log = load_pickle(self.logpath + name)
        return log

class ITrainAttrLog():
    def __init__(self, logpath='.log'):
        self.logpath = logpath
        self.loss_best = np.inf
        self.names_log = ['loss_best']

    def add_names_log(self, names_dict):
        self.names_log += names_dict

    def update_log_best(self, loss, name):
        if loss < self.loss_best:
            self.loss_best = loss
            save_pickle(self.get_log(), name, flag_show=False)

    def save_log(self, name='log.pkl'):
        save_pickle(self.get_log(), name, flag_show=False)

    def load_log(self, name='log.pkl', isSetAttr=True):
        log = load_pickle(name)
        if isSetAttr:
            for key, value in log.items():
                setattr(self, key, value)
        return log

    def get_log(self):
        log = {}
        for key in self.names_log:
            log[key] = getattr(self, key)

        return log

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_scheduler(optimizer, n_iters, lr, type='exp'):
    if type == 'exp':
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1e-5/lr)**(1.0/n_iters))
    elif type == 'cos':
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=1e-5)
    elif type == '1cycle':
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=n_iters,
            pct_start=0.1,  # 前10%升高
            anneal_strategy='cos',
            final_div_factor=1e3  # 最终 lr 很小
        )
    elif type == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            min_lr=1e-5)
    else:
        raise ValueError
    return sched

class IModel(nn.Module):
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def get_mod(self):
        mod_list, name_list = [], []
        for name, param in self.named_parameters():
            name_parts = name.split('.')  # 根据 name 定位到该参数所属的模块
            mod = self
            for attr in name_parts[:-1]:
                mod = getattr(mod, attr)
            param_name = name_parts[-1]

            mod_list.append(mod), name_list.append(param_name)

        self.mod_list, self.name_list = mod_list, name_list

    def set_param(self, new_param):
        pointer = 0
        for ii in range(len(self.mod_list)):
            mod, name = self.mod_list[ii], self.name_list[ii]
            p = getattr(mod, name)
            numel = p.numel()
            new_values = new_param[pointer:pointer + numel].view_as(p)
            pointer += numel

            self._del_set_attr(self.mod_list[ii], self.name_list[ii], new_values)

    def get_param(self):
        param_list = []
        for mod, name in zip(self.mod_list, self.name_list):
            p = getattr(mod, name)
            param_list.append(p.reshape(-1))    # flatten each parameter to 1D
        return torch.cat(param_list)            # concatenate all into a single 1D tensor

    def _del_set_attr(self, obj, attr_name, attr_val):
        delattr(obj, attr_name)
        setattr(obj, attr_name, attr_val)

class TestModel(IModel):
    def __init__(self):
        super(TestModel, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.get_mod()

    def forward(self, x):
        return self.f(x)

def inv_by_solve(A: torch.Tensor) -> torch.Tensor:
    n = A.shape[-1]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    I = I.expand(*A.shape[:-2], n, n)

    # A^{-1}
    A_inv = torch.linalg.solve(A, I)

    return A_inv

def woodbury_inverse_torch(A: torch.Tensor,
                           U: torch.Tensor,
                           C: torch.Tensor,
                           V: torch.Tensor = None) -> torch.Tensor:
    """
    计算 (A + U C V)^(-1)

    参数
    ----
    A : (..., n, n)
    U : (..., n, k)
    C : (..., k, k)
    V : (..., k, n)

    返回
    ----
    M_inv : (..., n, n)
        (A + U C V)^(-1)
    """

    if V is None:
        V = U.transpose(-1, -2)

    A_inv = inv_by_solve(A)

    # A^{-1} U
    A_inv_U = A_inv @ U

    # C^{-1}
    C_inv = inv_by_solve(C)

    # middle = C^{-1} + V A^{-1} U
    middle = C_inv + V @ A_inv_U

    # (A + UCV)^(-1) = A^{-1} - A^{-1}U middle^{-1} V A^{-1}
    M_inv = A_inv - A_inv_U @ torch.linalg.solve(middle, V @ A_inv)

    return M_inv

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

def flatten_jacobian_dict(J_dict, out_dim=None):
    vals = list(J_dict.values())

    if out_dim is None:
        return torch.cat([v.reshape(-1) for v in vals], dim=0)
    else:
        return torch.cat([v.reshape(out_dim, -1) for v in vals], dim=-1)


def flatten_param_var(var_list):
    return torch.cat([v.reshape(-1) for v in var_list], dim=0)

def Np2Tensor(x, device='cpu', dtype=torch.float32):
    if isinstance(x, Tensor):
        return x.to(device).to(dtype)

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device).to(dtype)

    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=dtype, device=device)

    return x

def ensure_2d(a):
    """
    输出统一为 (N, n)
    保持输入类型不变（ndarray / Tensor）

    支持：
        (n,)   -> (1, n)
        (N, n) -> 不变
    """

    is_numpy = isinstance(a, np.ndarray)
    is_tensor = isinstance(a, torch.Tensor)

    if not (is_numpy or is_tensor):
        raise TypeError(f"_ensure_2d: unsupported type {type(a)}")

    ndim = a.ndim

    if ndim == 1:
        if is_numpy:
            a = a[None, :]
        else:
            a = a.unsqueeze(0)

    elif ndim == 2:
        pass

    else:
        raise ValueError(f"_ensure_2d: unsupported shape {a.shape}")

    return a

def ensure_3d(a):
    """
    输出统一为 (N, n, d)
    保持输入类型不变（ndarray / Tensor）

    支持：
        (n,)      -> (1, n, 1)
        (n, d)    -> (1, n, d)
        (N, n, d) -> 不变
    """

    is_numpy = isinstance(a, np.ndarray)
    is_tensor = isinstance(a, torch.Tensor)

    if not (is_numpy or is_tensor):
        raise TypeError(f"_ensure_3d: unsupported type {type(a)}")

    ndim = a.ndim

    if ndim == 1:
        if is_numpy:
            a = a[None, :, None]
        else:
            a = a.unsqueeze(0).unsqueeze(-1)

    elif ndim == 2:
        if is_numpy:
            a = a[None, :, :]
        else:
            a = a.unsqueeze(0)

    elif ndim == 3:
        pass

    else:
        raise ValueError(f"_ensure_3d: unsupported shape {a.shape}")

    return a

def generate_grid(xmin1, xmax1, dx1, xmin2=None, xmax2=None, dx2=None,
                  device=None, dtype=torch.float32):
    """
    生成二维网格，并排成 [N, 2] 的形式。

    如果只输入一组范围，则默认两个维度使用相同范围。

    return:
        x_grid: [N, 2]
    """

    if xmin2 is None:
        xmin2 = xmin1
    if xmax2 is None:
        xmax2 = xmax1
    if dx2 is None:
        dx2 = dx1

    x1 = torch.arange(
        xmin1,
        xmax1 + dx1,
        dx1,
        device=device,
        dtype=dtype
    )

    x2 = torch.arange(
        xmin2,
        xmax2 + dx2,
        dx2,
        device=device,
        dtype=dtype
    )

    X1, X2 = torch.meshgrid(
        x1,
        x2,
        indexing="ij"
    )

    x_grid_traj = torch.stack(
        [X1.reshape(-1), X2.reshape(-1)],
        dim=1
    )

    return x_grid_traj, x1, x2

def save_pickle(data, file_name, flag_show=True):
    # 将data保存成pkl文件
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()
    if flag_show:
        print(file_name + ' save successfully')

def load_pickle(file_name, flag_show=True):
    # pkl文件读取
    f = open(file_name, 'rb+')
    data = pickle.load(f)
    f.close()
    if flag_show:
        print(file_name + ' load successfully')

    return data

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

def save_show(flag_save, flag_show, filename, fig, dpi=None):
    if flag_save:
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    if not flag_show:
        plt.close(fig)

def set_ylim(y, ratio=0, ax=None):
    """Set the ylim according to data y with a given epitaxial ratio"""
    y = y.ravel()
    ymin, ymax = y.min(), y.max()
    Dy = ymax - ymin
    if Dy != 0:
        if ax is None:
            plt.ylim([ymin - ratio*Dy, ymax + ratio*Dy])
        else:
            ax.set_ylim([ymin - ratio*Dy, ymax + ratio*Dy])

def merge_figures(fig1, fig2):

    new_fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_old, ax_new in zip([fig1.axes[0], fig2.axes[0]], axes):
        for line in ax_old.get_lines():
            ax_new.plot(line.get_xdata(), line.get_ydata())

        ax_new.set_title(ax_old.get_title())

    plt.tight_layout()

    return new_fig

def interpolate_u(t, u, dt):
    """
    u: (batch, len_t, m)
    """
    t_idx = (t / dt).long().clamp(0, u.shape[1] - 1)   # 找到最近的时间索引
    return u[..., t_idx, :]                                 # 提取对应的 u

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

def Tensor2list(x):
    return f'{list(map(lambda x: f"{x:.4f}", x.cpu().detach().numpy()))}'

def is_symmetric(x: torch.Tensor, tol: float = 1e-8) -> bool:
    """
    判断矩阵是否对称（支持 batch）

    参数
    ----
    x : (..., n, n)
    tol : 容忍误差

    返回
    ----
    bool
    """
    if x.ndim < 2:
        raise ValueError("Input must be at least 2D")

    if x.shape[-1] != x.shape[-2]:
        return False

    # 对称误差（Frobenius范数）
    diff = x - x.transpose(-1, -2)
    err = torch.norm(diff, dim=(-2, -1))

    return torch.all(err < tol).item()

def pca_torch(X, n_components=None, center=True):
    """
    用 torch 实现兼容批计算的 PCA（基于 SVD）

    参数
    ----
    X : torch.Tensor
        shape = (..., N, d)
        最后两维分别是样本维 N 和特征维 d
    n_components : int or None
        保留的主成分数，None 表示保留 min(N, d) 个
    center : bool
        是否沿样本维做中心化

    返回
    ----
    result : dict
        {
            "mean": shape (..., 1, d)
            "components": shape (..., k, d)
            "explained_variance": shape (..., k)
            "explained_variance_ratio": shape (..., k)
            "scores": shape (..., N, k)
            "X_centered": shape (..., N, d)
            "singular_values": shape (..., k)
        }
    """
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)

    if X.ndim < 2:
        raise ValueError(f"X 至少应为二维，当前 shape={tuple(X.shape)}")

    N, d = X.shape[-2], X.shape[-1]
    if N < 2:
        raise ValueError("PCA 至少需要两个样本")

    kmax = min(N, d)
    if n_components is None:
        k = kmax
    else:
        k = int(n_components)
        if not (1 <= k <= kmax):
            raise ValueError(f"n_components 应满足 1 <= k <= {kmax}")

    # ===== 1. 中心化 =====
    if center:
        mean = X.mean(dim=-2, keepdim=True)   # (..., 1, d)
        X_centered = X - mean
    else:
        mean = torch.zeros(*X.shape[:-2], 1, d, dtype=X.dtype, device=X.device)
        X_centered = X

    # ===== 2. Batched SVD =====
    # X_centered: (..., N, d)
    # U:  (..., N, r)
    # S:  (..., r)
    # Vh: (..., r, d)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    components = Vh[..., :k, :]                     # (..., k, d)
    singular_values = S[..., :k]                   # (..., k)

    # ===== 3. 主成分方差 =====
    explained_variance = (singular_values ** 2) / (N - 1)   # (..., k)
    all_explained_variance = (S ** 2) / (N - 1)             # (..., r)
    explained_variance_ratio = explained_variance / (
        all_explained_variance.sum(dim=-1, keepdim=True) + 1e-12
    )

    # ===== 4. 投影 =====
    # scores = X_centered @ components^T
    scores = X_centered @ components.transpose(-1, -2)      # (..., N, k)

    return {
        "mean": mean,
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "scores": scores,
        "X_centered": X_centered,
        "singular_values": singular_values,
    }

# region network data check
def inspect_stats(model):
    assert isinstance(model, torch.nn.Sequential)

    print("=== 参数统计 ===")
    for name, param in model.named_parameters():
        data = param.detach().cpu()

        print(f"{name}")
        print(f" mean={data.mean():.4e}, std={data.std():.4e}, "
              f"min={data.min():.4e}, max={data.max():.4e}")
        print("-" * 40)

def inspect_layers(model):
    assert isinstance(model, torch.nn.Sequential)

    print("=== 层级参数 ===")
    for i, layer in enumerate(model):
        print(f"\n[Layer {i}] {layer.__class__.__name__}")

        if hasattr(layer, 'weight') and layer.weight is not None:
            print(f" weight: shape={tuple(layer.weight.shape)}")

        if hasattr(layer, 'bias') and layer.bias is not None:
            print(f" bias:   shape={tuple(layer.bias.shape)}")

def inspect_values(model, max_elements=5):
    assert isinstance(model, torch.nn.Sequential)

    print("=== 参数数值（截断显示）===")
    for name, param in model.named_parameters():
        data = param.detach().cpu().flatten()
        snippet = data[:max_elements]

        print(f"{name}")
        print(f" shape={tuple(param.shape)}")
        print(f" values={snippet}")
        print("-" * 40)

def inspect_gradients(model):
    for name, p in model.named_parameters():
        if p.grad is None:
            print(name, 'grad=None')
        else:
            g = p.grad.detach()
            print(name, g.abs().mean().item(), g.abs().max().item())

def inspect_activations(model, x):
    a = x
    for i, layer in enumerate(model):
        a = layer(a)
        print(f'layer {i}: mean={a.mean().item():.4e}, std={a.std(unbiased=False).item():.4e}, '
              f'min={a.min().item():.4e}, max={a.max().item():.4e}')
# endregion

if __name__ == '__main__':
    testmodel = TestModel()
    param = nn.Parameter(torch.randn(testmodel.param_count()))
    testmodel.set_param(param)

    x = torch.linspace(-2, 2, 1000).view(-1, 1)
    y = torch.sin(x)

    optimizer = optim.Adam({param}, lr=1e-2)
    for i in range(2000):
        optimizer.zero_grad()
        loss = ((testmodel(x) - y)**2).mean()
        loss.backward()
        optimizer.step()
        print(loss.item())

    y_pre = testmodel(x)
    fig = plt.figure()
    plt.plot(x.detach().numpy(), y.detach().numpy(), label='y')
    plt.plot(x.detach().numpy(), y_pre.detach().numpy(), label='y_pre')
    plt.legend()

    plt.show()

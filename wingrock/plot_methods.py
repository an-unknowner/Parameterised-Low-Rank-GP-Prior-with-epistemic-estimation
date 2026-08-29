import numpy as np
from matplotlib import pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

from typing import List, Optional, Union, Tuple

# 设置全局字体
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 24
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Times New Roman'
# plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
# plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# plt.rcParams['text.usetex'] = True
def plot_single(
        x_data: Union[List, np.ndarray],
        y_data_list: Union[List[np.ndarray], np.ndarray],
        variance: Optional[Union[List, np.ndarray]] = None,
        names: Optional[List[str]] = None,
        truth: Optional[float] = None,
        colors: Optional[List[str]] = None,
        styles: Optional[List[str]] = None,
        x_range: Tuple[float, float] = None,
        y_range: Tuple[float, float] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
) -> plt.Axes:
    """
    绘制多条曲线图，支持方差阴影、自定义名称、颜色和样式

    参数:
    x_data: x轴数据，列表或numpy数组
    y_data_list: 一个或多个y轴数据，每个应与x_data长度相同
    variance: 可选，方差数据列表，每个元素应与对应的y_data长度相同
    names: 可选，曲线名称列表
    colors: 可选，曲线颜色列表
    styles: 可选，曲线样式列表
    ax: 可选，matplotlib的Axes对象，如果提供则在该Axes上绘图
    **kwargs: 其他传递给plot函数的参数

    返回:
    matplotlib的Axes对象，方便进一步自定义
    """
    # 确定曲线数量
    N = len(y_data_list)

    # 检查y_data_list中每个数组的长度是否与x_data一致
    x_len = len(x_data)
    for i, y_data in enumerate(y_data_list):
        if len(y_data) != x_len:
            raise ValueError(f"第{i + 1}个y数据长度({len(y_data)})与x数据长度({x_len})不匹配")

    # 检查可选参数是否存在且长度正确
    optional_params = {
        "variance": variance,
        "names": names,
        "styles": styles
    }

    for param_name, param_value in optional_params.items():
        if param_value is not None and len(param_value) != N:
            raise ValueError(f"{param_name}参数长度({len(param_value)})与y数据数量({N})不匹配")

    # 检查方差数据长度是否正确（如果提供了方差数据）
    if variance is not None:
        for i, var_data in enumerate(variance):
            if len(var_data) != x_len:
                raise ValueError(f"第{i + 1}个方差数据长度({len(var_data)})与x数据长度({x_len})不匹配")

    # 设置默认值
    if names is None:
        names = [f"曲线 {i + 1}" for i in range(N)]

    if colors is None:
        colors = [f"C{i}" for i in range(N)]  # 默认使用 matplotlib 循环色
    else:
        if len(colors) == 1:
            colors = colors * N  # 所有曲线同一个颜色
        elif len(colors) != N:
            raise ValueError(f"colors参数长度({len(colors)})应为1或{N}")

    if styles is None:
        styles = ['-'] * N  # 默认实线

    # 创建或使用传入的Axes对象
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制每条曲线
    for i in range(N):
        y_data = y_data_list[i]

        # 绘制方差阴影（如果提供了方差数据）
        if variance is not None:
            var_data = variance[i]
            ax.fill_between(
                x_data,
                np.array(y_data) - np.array(var_data),
                np.array(y_data) + np.array(var_data),
                color=colors[i],
                alpha=0.2
            )

        # 绘制曲线
        ax.plot(
            x_data,
            y_data,
            label=names[i],
            color=colors[i],
            linestyle=styles[i],
            linewidth=4,
            **kwargs
        )

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    if truth is not None:
        ax.axhline(truth, linestyle='dashed', color='red', label='pruning threshold', linewidth=4)

    # 添加图例
    ax.legend(loc='lower left', prop={'family': 'Times New Roman', 'size': 18})

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    return ax


def plot_bar(Ws,
             names=None,
             colors=None,
             variances=None,
             log_scale=False,
             ax=None):
    """
    绘制结果图
    - 如果没有 variance: 使用柱状图
    - 如果有 variance: 使用带误差棒的圆点图 (95% CI)

    参数:
    ----------
    Ws : array-like, shape (N,)
        每个点/柱子的值
    names : list of str
        每个点/柱子的标签
    colors : list of str
        每个点/柱子的颜色
    variances : array-like, shape (N,), optional
        每个点的方差，用于绘制95%置信区间
    ax : matplotlib.axes.Axes
        如果提供则在该坐标轴绘图，否则新建一个
    """
    """
    绘制结果图
    - 如果没有 variance: 使用柱状图
    - 如果有 variance: 使用带误差棒的圆点图 (95% CI)

    参数:
    ----------
    Ws : array-like, shape (N,)
        每个点/柱子的值
    names : list of str
        每个点/柱子的标签
    colors : list of str
        每个点/柱子的颜色
    variances : array-like, shape (N,), optional
        每个点的方差，用于绘制95%置信区间
    log_scale : bool, default=False
        是否使用对数坐标系 (y轴)
    ax : matplotlib.axes.Axes
        如果提供则在该坐标轴绘图，否则新建一个
    """
    Ws = np.array(Ws).flatten()
    Ws = np.abs(Ws)#*np.asarray([50,1,5,50,50])
    N = len(Ws)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    if names is not None:
        labels = names[:N]
        ax.set_xticklabels(labels, fontsize=20)
    item_colors = colors[:N]

    ax.set_xticks(range(N))

    ax.set_ylabel("Value", fontsize=20)

    if variances is not None:
        # 计算95% CI
        stds = np.sqrt(np.array(variances).flatten())
        ci95 = 1.96 * stds

        # 画圆点 + 误差棒
        for i, (w, c, ci) in enumerate(zip(Ws, item_colors, ci95)):
            for i, (w, c, ci) in enumerate(zip(Ws, item_colors, ci95)):
                ax.errorbar(i, w, yerr=ci,
                            fmt="o", markersize=12,  # 圆圈大一些
                            mfc="white", mec=c, mew=3,  # 圆圈边线更粗
                            ecolor="black", elinewidth=2.5, capsize=7, capthick=2)


            ax.text(i, w + ci * 1.05,
                    f"{w:.2f}±{ci:.2f}",
                    ha="center", va="bottom", fontsize=20)

        # 自动调整 y 轴范围
        ymax = max(Ws + ci95) * 1.5
        ymin = min(Ws - ci95)
        # 手动设置 y 轴范围
        if log_scale:
            ymin = max(ymin, 1e-3)  # 避免对数下溢
            ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_ylim(-0.2, max(Ws + ci95) * 2.0)  # 例如手动放大一倍

    else:
        # 画柱状图
        bars = ax.bar(range(N), Ws, color=item_colors, alpha=0.5,
                      edgecolor="black", linewidth=1.2)

        for bar, w in zip(bars, Ws):
            if abs(w) < 0.01:
                label = f"{w:.1e}"  # 科学计数法，1 位小数
            else:
                label = f"{w:.2f}"  # 普通小数，保留两位
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center", va="bottom", fontsize=20)

        if log_scale:
            ymin = max(min(Ws) * 0.8, 1e-3)
            ymax = max(Ws) * 1.5
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)

    ax.grid(True, linestyle="--", alpha=1.0)
    ax.tick_params(axis="y", width=2, length=6, labelsize=20)  # 刻度线加粗
    ax.spines["left"].set_linewidth(2)  # y轴主线加粗
    ax.spines["bottom"].set_linewidth(2)  # x轴主线也加粗（可选）

    return ax

def plot_multiple_subplots(
        x_data: Union[List, np.ndarray],
        *y_data_list: Union[np.ndarray, List[np.ndarray]],
        variance: Optional[List[np.ndarray]] = None,
        names: Optional[List[str]] = None,
        titles: Optional[List[str]] = None,
        axes: Optional[List[plt.Axes]] = None,
        colors: Optional[List[List[str]]] = None,
        truth: Optional[Union[List[float], np.ndarray]] = None,
        styles: Optional[List[List[str]]] = None,
        labels: Optional[List[List[str]]] = None,  # ✅ 新增参数，每个 y_data 对应的图例名称列表
        figsize: Tuple[int, int] = (10, 6),
        x_range: Tuple[float, float] = None,
        y_range: Tuple[float, float] = None,
        sharex: bool = False,
        sharey: bool = False,
        grid: bool = False,
        dt: float = None,
        **kwargs
) -> List[plt.Axes]:
    if len(y_data_list) == 0:
        raise ValueError("必须提供至少一个 y_data")

    x_len = len(x_data)

    # 处理 y_data_list，保证二维 (T, n_series)
    y_data_list_proc = []
    for y in y_data_list:
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        y_data_list_proc.append(y)

    # 子图数量 = 最大分量数
    num_series = max(y.shape[1] for y in y_data_list_proc)

    # 检查长度一致
    for y in y_data_list_proc:
        if len(y) != x_len:
            raise ValueError("所有 y 的长度必须和 x_data 一致")

    # 处理方差
    if variance is not None:
        if len(variance) != len(y_data_list_proc):
            raise ValueError("variance 数量必须与 y_data_list 对应")
        var_proc = []
        for v in variance:
            v = np.asarray(v)
            if v.ndim == 1:
                v = v[:, None]
            var_proc.append(v)
        variance = var_proc

    # 默认值
    if names is None:
        names = [f"Weight"] * num_series
    if titles is None:
        titles = [f"子图 {i + 1}" for i in range(num_series)]
    if colors is None:
        colors = [["C0"] for _ in range(len(y_data_list_proc))]
    if styles is None:
        styles = ['-'] * len(y_data_list_proc)

    # 默认 labels
    if labels is None:
        labels = [[f"Series {k + 1}" for k in range(y.shape[1])] for y in y_data_list_proc]

    if axes is None:
        fig, axes = plt.subplots(
            nrows=num_series,
            ncols=1,
            figsize=(figsize[0], figsize[1] * num_series / 2),
            sharex=sharex,
            sharey=sharey,
            squeeze=False
        )
    axes = axes.flatten()
    for ax in axes:
        ax.tick_params(labelbottom=True)

    # 绘制
    for j in range(num_series):
        ax = axes[j]
        for k, y in enumerate(y_data_list_proc):
            y_comp = y[:, j] if j < y.shape[1] else y[:, -1]

            # 颜色
            y_colors = colors[k] if colors else [f"C{k}"]
            if len(y_colors) == 1:
                color = y_colors[0]
            elif len(y_colors) == num_series:
                color = y_colors[j]
            else:
                raise ValueError(f"colors[{k}] 长度必须为1或{num_series}")

            # 样式
            y_styles = styles[k] if styles else '-'
            if len(y_styles) == 1:
                style = y_styles[0]
            elif len(y_styles) == num_series:
                style = y_styles[j]
            else:
                raise ValueError(f"styles[{k}] 长度必须为1或{num_series}")

            y_labels = labels[k] if labels else '-'
            if len(y_labels) == 1:
                label = y_labels[0]
            elif len(y_labels) == num_series:
                label = y_labels[j]
            else:
                raise ValueError(f"y_labels[{k}] 长度必须为1或{num_series}")

            # 方差
            if variance is not None:
                var_comp = variance[k][:, j] if j < variance[k].shape[1] else variance[k][:, -1]
                ax.fill_between(
                    x_data,
                    y_comp - np.sqrt(var_comp) * 1.96,
                    y_comp + np.sqrt(var_comp) * 1.96,
                    color=color, alpha=0.1
                )

            ax.plot(x_data, y_comp, color=color, linestyle=style, label=label, linewidth=3, alpha = 0.5, **kwargs)

        if x_range is not None:
            ax.set_xlim(x_range)
        if y_range is not None:
            ax.set_ylim(y_range)

        if truth is not None:
            ax.axhline(truth[j], linestyle='dashed', color='k', label='True')
        ax.legend(loc='lower left', prop={'family': 'Times New Roman', 'size': 20})

        ax.set_title(titles[j], fontname='Times New Roman', fontsize=28)
        ax.set_ylabel(names[j], fontname='Times New Roman', fontsize=28)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)

    # x轴标签
    if dt is not None:
        if sharex:
            axes[-1].set_xlabel(f"time ({dt}s)", fontname='Times New Roman', fontsize=28)
        else:
            for ax in axes:
                ax.set_xlabel(f"time ({dt}s)", fontname='Times New Roman', fontsize=28)

    if sharex:
        axes[-1].set_xlabel(f"time (s)", fontname='Times New Roman', fontsize=28)
    else:
        for ax in axes:
            ax.set_xlabel(f"time (s)", fontname='Times New Roman', fontsize=28)

    plt.tight_layout()
    return axes

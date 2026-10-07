"""生成内训幻灯片中使用的图像。

这些图不是手绘示意图，而是对一组合成数据真实执行「最小二乘拟合、梯度下降训练、
多项式拟合」之后得到的结果，用来在课堂上展示每一步的实际输出。
运行方式：python3 make_figures.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Hiragino Sans GB"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(11)

ACCENT = "#285E8E"
GREEN = "#12B76A"
RED = "#D92D20"
ORANGE = "#B54708"
INK = "#1F2A37"
MUTED = "#667085"


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", name)


def clean(ax, title=None, xlabel=None, ylabel=None, legend=False):
    for spine in ax.spines.values():
        spine.set_color("#C8CDD4")
    if title:
        ax.set_title(title, fontsize=15, color=INK, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, color=MUTED)
    ax.tick_params(labelsize=11, colors=MUTED)
    if legend:
        leg = ax.legend(fontsize=12, frameon=False)
        for text in leg.get_texts():
            text.set_color(MUTED)


# ---------------------------------------------------------------- 数据
# 一台小车的电机标定数据：给一个 PWM 指令，量一次实际速度
PWM = np.linspace(1000.0, 2000.0, 18)
SPEED_TRUE = 0.40 * (PWM - 1000.0) + 30.0
SPEED = SPEED_TRUE + rng.normal(0.0, 12.0, PWM.size)

W_FIT, B_FIT = np.polyfit(PWM, SPEED, 1)
B_CENTER = W_FIT * 1500.0 + B_FIT          # 写成以 1500 为中心的截距，便于阅读
SPEED_PRED = W_FIT * PWM + B_FIT
MSE = float(np.mean((SPEED_PRED - SPEED) ** 2))

print("标定数据（PWM, 实测速度 mm/s）：")
for p, s in zip(PWM, SPEED):
    print(f"  {p:.0f}  {s:6.1f}")
print(f"最小二乘拟合：斜率 {W_FIT:.4f}，中心截距 {B_CENTER:.1f}，MSE {MSE:.1f}")


# ---------------------------------------------------------------- 1 散点与拟合直线
fig, ax = plt.subplots(figsize=(9.0, 4.6))
ax.scatter(PWM, SPEED, s=46, color=ACCENT, zorder=3, label="实测样本")
xs = np.linspace(1000, 2000, 200)
ax.plot(xs, W_FIT * xs + B_FIT, color=RED, lw=2.4, zorder=2, label="拟合直线")
ax.set_xlim(960, 2040)
ax.set_ylim(0, 480)
clean(ax, None, "PWM 指令", "实测速度 (mm/s)", legend=True)
ax.text(1005, 410, f"预测速度 = {W_FIT:.3f} × (PWM - 1500) + {B_CENTER:.1f}", fontsize=15, color=RED)
save(fig, "fig_scatter_line.png")


# ---------------------------------------------------------------- 2 误差与 MSE
fig, ax = plt.subplots(figsize=(9.0, 4.6))
for x0, y0, y1 in zip(PWM, SPEED, SPEED_PRED):
    ax.plot([x0, x0], [y0, y1], color="#98A2B3", lw=1.2, zorder=1)
ax.scatter(PWM, SPEED, s=46, color=ACCENT, zorder=3, label="实测值")
ax.plot(xs, W_FIT * xs + B_FIT, color=RED, lw=2.4, zorder=2, label="模型预测值")
ax.set_xlim(960, 2040)
ax.set_ylim(0, 480)
clean(ax, None, "PWM 指令", "实测速度 (mm/s)", legend=True)
ax.text(1005, 410, f"均方误差 MSE = {MSE:.1f}\n（每一段竖线的长度就是一个误差）",
        fontsize=15, color=INK)
save(fig, "fig_error.png")


# ---------------------------------------------------------------- 3 梯度下降
# 互动用图：两组候选参数，让学生先猜哪一组的 MSE 更小
W_A, B_A = 0.20, 120.0
W_B, B_B = float(W_FIT), float(B_CENTER)
MSE_A = float(np.mean((W_A * (PWM - 1500.0) + B_A - SPEED) ** 2))
MSE_B = float(np.mean((W_B * (PWM - 1500.0) + B_B - SPEED) ** 2))

fig, ax = plt.subplots(figsize=(9.0, 4.6))
ax.scatter(PWM, SPEED, s=46, color=ACCENT, zorder=3, label="实测样本")
ax.plot(xs, W_A * (xs - 1500.0) + B_A, color=ORANGE, lw=2.4, zorder=2,
        label=f"A：w = {W_A:.2f}, b = {B_A:.0f}")
ax.plot(xs, W_B * (xs - 1500.0) + B_B, color=GREEN, lw=2.4, zorder=2,
        label=f"B：w = {W_B:.2f}, b = {B_B:.0f}")
ax.set_xlim(960, 2040)
ax.set_ylim(0, 480)
clean(ax, None, "PWM 指令", "实测速度 (mm/s)", legend=True)
save(fig, "fig_two_lines.png")
print(f"候选参数 A 的 MSE {MSE_A:.1f}，B 的 MSE {MSE_B:.1f}")


# ---------------------------------------------------------------- 4 梯度下降
# 为了把曲线画清楚，先把特征归一化到 [-1, 1]，并固定 b 只调 w
X_NORM = (PWM - 1500.0) / 500.0
B_STAR = float(SPEED.mean())
W_STAR = float(np.sum(X_NORM * (SPEED - B_STAR)) / np.sum(X_NORM ** 2))
CURVATURE = 2.0 * float(np.mean(X_NORM ** 2))


def loss(w):
    w = np.asarray(w, dtype=float)
    return np.mean((w[..., None] * X_NORM + B_STAR - SPEED) ** 2, axis=-1)


def descend(eta, steps, start=120.0):
    path = [start]
    for _ in range(steps):
        path.append(path[-1] - eta * CURVATURE * (path[-1] - W_STAR))
    return np.array(path)


GRID = np.linspace(20.0, 360.0, 400)
CURVE = loss(GRID)

path_mid = descend(0.5, 8, start=40.0)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
axes[0].plot(GRID, CURVE, color=ACCENT, lw=2.2)
axes[0].plot(path_mid, loss(path_mid), "o-", color=RED, ms=6, lw=1.4, zorder=3)
axes[0].annotate("起点", xy=(path_mid[0], loss(path_mid[0])), xytext=(path_mid[0] + 30, loss(path_mid[0]) + 900),
                 fontsize=13, color=RED)
axes[0].annotate("最低点", xy=(W_STAR, loss(W_STAR)), xytext=(W_STAR + 12, loss(W_STAR) + 2200),
                 fontsize=13, color=GREEN)
clean(axes[0], "损失随参数 w 变化", "参数 w", "损失 L")
axes[1].plot(np.arange(path_mid.size), loss(path_mid), "o-", color=RED, ms=6, lw=1.8)
clean(axes[1], "损失随迭代次数下降", "迭代次数", "损失 L")
save(fig, "fig_gradient_descent.png")


# ---------------------------------------------------------------- 4 学习率
settings = [
    (0.05, "η = 0.05 · 收敛很慢", RED),
    (0.80, "η = 0.80 · 几步到位", GREEN),
    (3.20, "η = 3.20 · 震荡发散", RED),
]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
for ax, (eta, title, color) in zip(axes, settings):
    path = descend(eta, 8)
    ax.plot(GRID, CURVE, color="#C7D9EA", lw=2.0)
    inside = (path >= 20.0) & (path <= 360.0)
    ax.plot(path[inside], loss(path[inside]), "o-", color=color, ms=6, lw=1.4, zorder=3)
    ax.set_xlim(20, 360)
    ax.set_ylim(0, 2600)
    clean(ax, title, "参数 w", "损失 L")
fig.subplots_adjust(wspace=0.26)
save(fig, "fig_learning_rate.png")


# ---------------------------------------------------------------- 5 过拟合与欠拟合
x_train = np.sort(rng.uniform(-1.0, 1.0, 14))
def target(x):
    return 0.62 * np.sin(3.0 * x)

y_train = target(x_train) + rng.normal(0.0, 0.09, x_train.size)
x_val = np.linspace(-1.0, 1.0, 60)
y_val = target(x_val) + rng.normal(0.0, 0.09, x_val.size)
curve_x = np.linspace(-1.05, 1.05, 300)

panels = [(1, "1 次多项式：欠拟合"), (3, "3 次多项式：合适"), (9, "9 次多项式：过拟合")]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
for ax, (degree, title) in zip(axes, panels):
    coeffs = np.polyfit(x_train, y_train, degree)
    train_err = float(np.mean((np.polyval(coeffs, x_train) - y_train) ** 2))
    val_err = float(np.mean((np.polyval(coeffs, x_val) - y_val) ** 2))
    ax.plot(curve_x, np.polyval(coeffs, curve_x), color=ACCENT, lw=2.2, zorder=2)
    ax.scatter(x_train, y_train, s=34, color=RED, zorder=3, label="训练样本")
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlim(-1.15, 1.15)
    clean(ax, title, "输入特征 x", "输出 y")
    ax.text(-1.05, -1.42, f"训练误差 {train_err:.3f}\n验证误差 {val_err:.3f}", fontsize=12, color=MUTED)
fig.subplots_adjust(wspace=0.26)
save(fig, "fig_overfit.png")

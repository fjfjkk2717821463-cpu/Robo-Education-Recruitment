"""生成内训幻灯片中使用的图像。

这些图不是手绘示意图，而是对一张合成画面真实执行颜色阈值、形态学、轮廓与质心计算
之后得到的中间结果，用来在课堂上展示每一步的实际输出。
运行方式：python3 make_figures.py
"""

from collections import deque
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

matplotlib.rcParams["font.family"] = ["Hiragino Sans GB"]
matplotlib.rcParams["axes.unicode_minus"] = False

H, W = 360, 640
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(7)

YY, XX = np.mgrid[0:H, 0:W].astype(float)


def disc(cx, cy, r):
    return (XX - cx) ** 2 + (YY - cy) ** 2 <= r * r


def make_frame():
    base = np.zeros((H, W, 3), float)
    base[..., 0] = 0.87 - 0.10 * (XX / W)
    base[..., 1] = 0.89 - 0.08 * (XX / W)
    base[..., 2] = 0.91 - 0.05 * (XX / W)
    frame = base + rng.normal(0, 0.010, (H, W, 3))

    ball = disc(415, 150, 46)
    d = ((XX - 395) ** 2 + (YY - 130) ** 2) / (95.0 ** 2)
    shade = np.clip(1.0 - 0.55 * np.sqrt(np.clip(d, 0, 1)), 0.45, 1.0)
    lit = np.stack([0.85 * shade, 0.17 * shade, 0.14 * shade], -1)
    frame[ball] = lit[ball]

    frame[disc(115, 258, 32)] = np.array([0.18, 0.36, 0.78])

    # 一个极小的红色碎屑（会被开运算去掉）与一个偏大的同色干扰物（要靠面积筛选去掉）
    frame[disc(196, 84, 2)] = np.array([0.72, 0.31, 0.28])
    frame[disc(548, 300, 7)] = np.array([0.72, 0.30, 0.27])
    return np.clip(frame, 0, 1)


def in_range_red(frame, s_min=0.45, v_min=0.32):
    hsv = rgb_to_hsv(frame)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = (hue >= 0.93) | (hue <= 0.06)
    return hue_ok & (sat >= s_min) & (val >= v_min)


def _morph(mask, k, op, fill):
    r = k // 2
    padded = np.pad(mask, r, mode="constant", constant_values=fill)
    out = np.full_like(mask, fill, dtype=bool)
    for dy in range(k):
        for dx in range(k):
            out = op(out, padded[dy:dy + H, dx:dx + W])
    return out


def dilate(mask, k=9):
    return _morph(mask, k, np.maximum, False)


def erode(mask, k=9):
    return _morph(mask, k, np.minimum, True)


def component_boxes(mask, min_area=1):
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y0, x0 in np.argwhere(mask):
        if seen[y0, x0]:
            continue
        queue = deque([(y0, x0)])
        seen[y0, x0] = True
        comp = []
        while queue:
            y, x = queue.popleft()
            comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(comp) >= min_area:
            ys = np.array([p[0] for p in comp])
            xs = np.array([p[1] for p in comp])
            boxes.append({
                "area": len(comp),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "centroid": (float(xs.mean()), float(ys.mean())),
            })
    boxes.sort(key=lambda b: -b["area"])
    return boxes


def clean_axes(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C8CDD4")
    if title:
        ax.set_title(title, fontsize=15, color="#1F2A37", pad=10)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote", name)


frame = make_frame()
mask_raw = in_range_red(frame)
mask_eroded = erode(mask_raw, 7)
mask_opened = dilate(mask_eroded, 7)
boxes_raw = component_boxes(mask_raw)
boxes_open = component_boxes(mask_opened)
MIN_AREA = 1000
target = next(b for b in boxes_open if b["area"] >= MIN_AREA)


# 1) 图像就是矩阵
fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.imshow(frame)
ax.set_xlim(-8, W + 8)
ax.set_ylim(H + 8, -8)
clean_axes(ax)
ax.annotate("", xy=(W - 40, H - 22), xytext=(40, H - 22),
            arrowprops=dict(arrowstyle="<->", color="#285E8E", lw=1.7))
ax.text(W / 2, H - 32, "宽度 W = 640 像素", ha="center", va="bottom", fontsize=14, color="#285E8E")
ax.annotate("", xy=(26, H - 70), xytext=(26, 70),
            arrowprops=dict(arrowstyle="<->", color="#285E8E", lw=1.7))
ax.text(38, H / 2, "高度 H = 360", ha="center", va="center", fontsize=14,
        color="#285E8E", rotation=90)
ax.add_patch(plt.Rectangle((408, 143), 14, 14, fill=False, ec="#111827", lw=2.2))
ax.annotate("一个像素 = 3 个数 (B, G, R)",
            xy=(406, 146), xytext=(300, 26), fontsize=14, color="#111827",
            ha="left", arrowprops=dict(arrowstyle="-", color="#111827", lw=1.2))
save(fig, "fig_frame_matrix.png")


# 2) HSV 三个通道
hsv = rgb_to_hsv(frame)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
for ax, data, name in zip(axes, [hsv[..., 0], hsv[..., 1], hsv[..., 2]],
                          ["H 色调（是什么颜色）", "S 饱和度（颜色有多纯）", "V 明度（有多亮）"]):
    ax.imshow(data, cmap="gray", vmin=0, vmax=1)
    clean_axes(ax, name)
fig.subplots_adjust(wspace=0.12)
save(fig, "fig_hsv_channels.png")


# 3) 阈值分割
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
axes[0].imshow(frame)
clean_axes(axes[0], "① 摄像头拍到的一帧")
axes[1].imshow(mask_raw, cmap="gray")
clean_axes(axes[1], "② 红色阈值分割后的掩膜")
fig.subplots_adjust(wspace=0.1)
save(fig, "fig_mask.png")


# 4) 形态学去噪
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))
for ax, data, name in zip(axes, [mask_raw, mask_eroded, mask_opened],
                          ["原掩膜：目标 + 两处杂点", "腐蚀：小杂点消失", "再膨胀：目标恢复"]):
    ax.imshow(data, cmap="gray")
    clean_axes(ax, name)
fig.subplots_adjust(wspace=0.1)
save(fig, "fig_morphology.png")


# 5) 轮廓与面积筛选
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
axes[0].imshow(mask_opened, cmap="gray")
clean_axes(axes[0], "去噪后的掩膜")
axes[1].imshow(frame)
for box in boxes_open:
    x0, y0, x1, y1 = box["bbox"]
    keep = box["area"] >= MIN_AREA
    color = "#12B76A" if keep else "#D92D20"
    axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=2.2))
    axes[1].text(x0, y0 - 10, f"{box['area']} px", fontsize=13, color=color)
clean_axes(axes[1], "面积筛选：只保留大目标")
fig.subplots_adjust(wspace=0.1)
save(fig, "fig_contours.png")


# 6) 目标中心与水平偏差
cx, cy = target["centroid"]
fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.imshow(frame)
ax.set_xlim(-8, W + 8)
ax.set_ylim(H + 8, -8)
x0, y0, x1, y1 = target["bbox"]
ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#12B76A", lw=2.0))
ax.plot([cx], [cy], marker="+", ms=22, mew=3.5, color="#F79009")
ax.axvline(W / 2, color="#285E8E", ls="--", lw=1.8)
ax.annotate("", xy=(cx, 336), xytext=(W / 2, 336),
            arrowprops=dict(arrowstyle="<->", color="#D92D20", lw=2.2))
ax.text((cx + W / 2) / 2, 348, "水平偏差 = 目标横坐标 - 画面中心横坐标",
        ha="center", va="top", fontsize=14, color="#D92D20")
ax.annotate("目标中心 (x_target, y_target)", xy=(cx + 8, cy - 8), xytext=(474, 62),
            fontsize=14, color="#B54708",
            arrowprops=dict(arrowstyle="-", color="#B54708", lw=1.2))
ax.text(W / 2 - 14, 316, "画面中心 x_center", fontsize=14, color="#285E8E", ha="right")
clean_axes(ax)
save(fig, "fig_center_error.png")


print()
print("阈值分割后检测到的轮廓（面积从大到小）：")
for b in boxes_raw:
    print(f"  面积 {b['area']:>6} px   外接框 {b['bbox']}")
print(f"面积筛选后保留目标：面积 {target['area']} px，质心 ({cx:.1f}, {cy:.1f})")

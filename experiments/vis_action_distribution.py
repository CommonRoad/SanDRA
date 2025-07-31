import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

# --- TUM color palette ---
class TUMcolor(tuple, Enum):
    TUMblue = (0, 101 / 255, 189 / 255)
    TUMlightblue = (100/255, 160/255, 200/255)
    TUMgreen = (162 / 255, 173 / 255, 0)
    TUMyellow = (254 / 255, 215 / 255, 2 / 255)
    TUMgrey = (153 / 255, 153 / 255, 153 / 255)
    TUMred = (227 / 255, 27 / 255, 35 / 255)
    TUMdarkred = (139 / 255, 0, 0)
    TUMgray = (156 / 255, 157 / 255, 159 / 255)
    TUMdarkgray = (88 / 255, 88 / 255, 99 / 255)
    TUMorange = (227 / 255, 114 / 255, 34 / 255)
    TUMdarkblue = (0, 82 / 255, 147 / 255)
    TUMwhite = (1, 1, 1)
    TUMblack = (0, 0, 0)
    TUMlightgray = (217 / 255, 218 / 255, 219 / 255)
    TUMdarkyellow = (203 / 255, 171 / 255, 1 / 255)
    TUMdarkgreen = (125 / 255, 146 / 255, 42 / 255)

# --- Labels and Colors ---
labels = ['FASTER', 'SLOWER', 'IDLE', 'LANE_RIGHT', 'LANE_LEFT']
colors = [
    TUMcolor.TUMgreen.value,
    TUMcolor.TUMdarkgreen.value,
    TUMcolor.TUMyellow.value,
    TUMcolor.TUMblue.value,
    TUMcolor.TUMlightblue.value
]

# --- Data ---
dilu      = [0.1450, 0.4560, 0.3749, 0.0082, 0.0158]
ours_set  = [0.2304, 0.4079, 0.2673, 0.0243, 0.0702]
ours_ml   = [0.2085, 0.3029, 0.2132, 0.1101, 0.1653]

methods = ['Ours + Set', 'Ours + ML', 'DiLu', ]
data = [ours_set, ours_ml, dilu]

# --- 1. Stacked Bar Plot ---
def plot_stacked_bar():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))

    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [d[i] for d in data]
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.3)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_xticklabels(methods, rotation=90)  # Rotate tick labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3)
    plt.tight_layout()
    plt.show()

def plot_stacked_bar_horizontal():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    y = np.arange(len(methods))
    left = np.zeros(len(methods))

    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [d[i] for d in data]
        ax.barh(y, values, left=left, label=label, color=color, height=0.7)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)
    plt.tight_layout()
    plt.show()

# --- 2. Grouped Bar Plot ---
def plot_grouped_bar():
    fig, ax = plt.subplots(figsize=(8, 3))
    bar_width = 0.25
    x = np.arange(len(labels))

    for i, (method, values, offset) in enumerate(zip(methods, data, [-bar_width, 0, bar_width])):
        ax.bar(x + offset, values, width=bar_width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel('Proportion')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- 3. Radar Plot ---
def plot_radar_chart():
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    for i, method_data in enumerate(data):
        values = method_data + method_data[:1]
        ax.plot(angles, values, label=methods[i], linewidth=2)
        ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# --- 4. Dot Plot ---
def plot_dot_plot():
    fig, ax = plt.subplots(figsize=(8, 3))

    for i, label in enumerate(labels):
        for j, method in enumerate(methods):
            ax.plot(data[j][i], i + j * 0.2, 'o',
                    label=method if i == 0 else "",
                    color=colors[i])

    ax.set_yticks(np.arange(len(labels)) + 0.2)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Proportion")
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- Run All Plots ---
plot_stacked_bar_horizontal()
plot_stacked_bar()
plot_grouped_bar()
plot_radar_chart()
plot_dot_plot()

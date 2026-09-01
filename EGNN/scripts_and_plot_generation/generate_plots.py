import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

#replace with your own paths to the log/result files
file_map = {
    "egnn5l":          "/mnt/c/Users/adeli/Downloads/std_a100_1866383.out",
    "egnn7l":          "/mnt/c/Users/adeli/Downloads/std_a1007L_1892681.out",
    "hyegnn_5_5_128":  "/mnt/c/Users/adeli/Downloads/hybrid_5v5_1890527.out",
    "hyegnn_5_5_64":   "/mnt/c/Users/adeli/Downloads/hybrid_5p5_nf64_bs96_1877151.out",
}

raw = {}
for name, path in file_map.items():
    epochs, maes, mses = [], [], []
    current_epoch = None
    with open(path) as f:
        for line in f:
            m = re.match(r"Val loss:.*epoch (\d+)", line)
            if m:
                current_epoch = int(m.group(1))
            m2 = re.match(r"\s+Test MAE: ([\d.]+)\s+\t MSE: ([\d.]+)", line)
            if m2 and current_epoch is not None:
                epochs.append(current_epoch)
                maes.append(float(m2.group(1)))
                mses.append(float(m2.group(2)))
                current_epoch = None
    raw[name] = {"epochs": np.array(epochs), "mae": np.array(maes), "mse": np.array(mses)}

CURVES = [
    ("egnn5l",         "EGNN(5)",          "#FF6D00", "-",  "^",  1.8),
    ("egnn7l",         "EGNN(7)",          "#30b030", "-",  "D",  1.8),
    ("hyegnn_5_5_128", "HyEGNN(5,5,128)",  "#e03525", "-",  "s",  2.2),
    ("hyegnn_5_5_64",  "HyEGNN(5,5,64)",   "#1a6fd4", "-",  "o",  2.2),
]

min_mse_info = {}
for key, *_ in CURVES:
    d = raw[key]
    best_i = d["mse"].argmin()
    min_mse_info[key] = {"epoch": d["epochs"][best_i], "mse": d["mse"][best_i], "mae": d["mae"][best_i]}

label_y_frac = {
    "egnn5l":         0.90,
    "egnn7l":         0.78,
    "hyegnn_5_5_128": 0.66,
    "hyegnn_5_5_64":  0.54,
}

panels = [
    ("mse", "MSE (log scale)", "(a) MSE over epochs", 0.002, 0.15, 0.01, 0.082, 0.0018, [0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15]),
    ("mae", "MAE (log scale) [mEv]", "(b) MAE over epochs", 25, 300, 25, 300, None, [30, 40, 50, 70, 100, 150, 200, 300]),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
fig.patch.set_facecolor('white')

for ax, (metric, ylabel, title, ymin_fixed, ymax_fixed, step, label_max, ylim_min, custom_ticks) in zip(axes, panels):
    for key, label, color, ls, marker, lw in CURVES:
        d = raw[key]
        eps = d["epochs"].copy()
        vals = d[metric].copy()
        mask_clip = eps >= 0
        eps = eps[mask_clip]; vals = vals[mask_clip]
        if len(eps) > 100:
            mask = ((eps-1) % 25 == 0)
            eps = eps[mask]; vals = vals[mask]
        # Shift epochs by +1 so the first point is at epoch 1, not 0,
        # ensuring no curve touches the y-axis.
        eps = eps + 1
        if metric == "mae":
            vals = vals * 1000
        kwargs = dict(color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)
        if marker:
            kwargs.update(marker=marker, markersize=4, markevery=3)
        print(vals.max())
        ax.semilogy(eps, vals, **kwargs)

    ax.set_yscale('log')
    ax.set_ylim(ylim_min if ylim_min is not None else ymin_fixed, None)

    if custom_ticks is not None:
        ticks = np.array(custom_ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:g}' for t in ticks], fontsize=8.5)
    else:
        ticks = np.round(np.arange(ymin_fixed, ymax_fixed + step * 0.01, step), 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:.3f}' if t <= label_max + 1e-9 or abs(t - ticks[-1]) < 1e-9 else '' for t in ticks], fontsize=8.5)
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    log_ymin = np.log10(ymin_fixed)
    log_ymax = np.log10(ymax_fixed)
    log_range = log_ymax - log_ymin

    for key, label, color, ls, marker, lw in CURVES:
        info = min_mse_info[key]
        ep = info["epoch"] + 1  # also shift the vertical lines by +1
        y_label = 10 ** (log_ymin + label_y_frac[key] * log_range)
        vline_ls = '-.'
        ax.axvline(ep, color=color, linewidth=1.2, linestyle=vline_ls, alpha=0.75, zorder=2)
        ax.text(ep, y_label, label,
                fontsize=8, color=color, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color,
                          alpha=0.9, linewidth=0.8))

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=7)
    # Start x-axis at 1 so curves don't intersect the y-axis
    ax.set_xlim(-10, 1005)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle=':', linewidth=0.6, alpha=0.5, which='major')
    ax.tick_params(labelsize=9.5)
    plt.setp(ax.get_xticklabels(), fontweight='normal')

handles, labels = axes[0].get_legend_handles_labels()
handles.append(Line2D([0], [0], color='gray', linewidth=1.2, linestyle='-.'))
labels.append("Epoch at minimum test MSE")
fig.legend(handles, labels, loc='lower center', ncol=5,
           fontsize=10, framealpha=0.9, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, -0.02), prop={'weight': 'normal'})


plt.tight_layout(pad=1.4)
plt.subplots_adjust(bottom=0.18)
plt.savefig("plots.png", dpi=250, bbox_inches='tight', facecolor='white')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
matplotlib.rcParams['mathtext.default'] = 'bf'
 
# Data Definitions
locations = [
    "Wire", "Solder joint (S1)", "Solder joint (S2)", "Gilding (S1)", "Gilding (S2)",
    "Gilding (S3)", "Gilding (S4)", "Gilding (S5)", "Gilding (S6)", "Gilding (S7)"
]
 
labels = [
    "Wire",
    "Solder joint (ROI 1)", "Solder joint (ROI 2)",
    "Gilding #1", "Gilding #2", "Gilding #3", "Gilding #4", "Gilding #5", "Gilding #6", "Gilding #7"
]
 
markers = ['o', 'D', 'D', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
markersize = [17, 15, 15, 15, 15, 15, 15, 15, 15, 15]
markers_dict = {"Wire": "o", "Solder joint": "D", "Gilding": "X"}
legend_markersizes = {"Wire": 13, "Solder joint": 12, "Gilding": 13}
 
groups = ["Wire", "Solder joint", "Solder joint", "Gilding", "Gilding", "Gilding", "Gilding", "Gilding", "Gilding", "Gilding"]
 
# Simulated Values
sim_values = [2.43, 1.95, 1.87, 4.4333, 4.3595, 3.5397, 5.4582, 5.3108, 2.7771, 2.6054]
# Run-to-run sample standard deviations over the 10 MC runs.
sim_errors = [0.01, 0.01, 0.01, 0.0139, 0.0179, 0.0227, 0.0212, 0.0283, 0.0124, 0.0107]
 
# nCT-derived ranges
exp_wire = (2.3, 2.5)
exp_weld = (1.9, 2.1)
exp_gilding = (3.1, 3.8)   # observed nCT min/max for the gilding ROI
 
exp_wire_error = 0.02
exp_weld_error = 0.02
exp_gilding_error = 0.17   # 1-sigma on the bound, consistent with the inversion
 
# Styling Configuration
CONFIG = {
    "title_size": 18,
    "label_size": 16,
    "spine_linewidth": 1.5,
    "tick_size": 18,
    "line_width": 2.5,
    "grid_alpha": 0.5,
    "label_pad": 10.0,
    "font_weight": 'bold',
    "dpi": 300,
    'tick_width': 2,
    'tick_length': 7,
    'tick_labelsize': 14,
}
 
colors = {"Wire": "black", "Solder joint": "orangered", "Gilding": "dodgerblue"}
 
 
def generate_attenuation_plot(savepath="fig_composition_response.png"):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=CONFIG['dpi'])
    x = np.arange(len(locations))
 
    # nCT-derived ranges, with a +/-1 sigma band on each bound.
    #    (x0, width, (lower, upper), sigma)
    spans = [
        (-0.4, 0.8, exp_wire, exp_wire_error),      # Wire
        (0.6, 1.8, exp_weld, exp_weld_error),       # Solder joint
        (2.6, 6.8, exp_gilding, exp_gilding_error), # Gilding
    ]
    for x0, width, (lo, hi), err in spans:
        ax.add_patch(Rectangle((x0, lo), width, hi - lo, facecolor='lightgray',
                               hatch='//', edgecolor='dimgray', alpha=0.45,
                               zorder=1))
        for edge in (lo, hi):
            ax.add_patch(Rectangle((x0, edge - err), width, 2 * err,
                                   facecolor='dimgray', alpha=0.25,
                                   edgecolor='none', zorder=1))
 
    # Simulated points, with black marker edges.
    for i, group in enumerate(groups):
        ax.errorbar(x[i], sim_values[i], yerr=sim_errors[i], fmt=markers[i],
                    color=colors[group], ecolor=colors[group],
                    markersize=markersize[i], capsize=5, capthick=2,
                    elinewidth=CONFIG['line_width'], markeredgewidth=2,
                    markeredgecolor='black', zorder=5)
 
    # Legend (built explicitly so the range patches appear once each).
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor='lightgray', hatch='//',
                  edgecolor='dimgray', alpha=0.6),
        Rectangle((0, 0), 1, 1, facecolor='dimgray', alpha=0.35,
                  edgecolor='none'),
    ]
    legend_labels = ['nCT-derived range', r'Bound $\pm 1\sigma$']
    for group, color in colors.items():
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=markers_dict[group],
                          linestyle='None', markersize=legend_markersizes[group],
                          markeredgecolor='black', markeredgewidth=1.5))
        legend_labels.append(group)
 
    # Styling and labels
    ax.set_ylabel(r"Attenuation Coefficient $\Sigma_{MC}$ (cm$^{-1}$)",
                  fontsize=CONFIG['label_size'],
                  fontweight=CONFIG['font_weight'],
                  labelpad=CONFIG['label_pad'])
    ax.set_xlabel("Sample index", fontsize=CONFIG['label_size'],
                  fontweight=CONFIG['font_weight'],
                  labelpad=CONFIG['label_pad'])
 
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right',
                       fontsize=CONFIG['tick_labelsize'], fontweight='bold')
 
    ax.grid(True, axis='y', linestyle='--', alpha=CONFIG['grid_alpha'], zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(CONFIG['spine_linewidth'])
 
    ax.tick_params(axis='both', which='both', labelsize=CONFIG['tick_labelsize'],
                   width=CONFIG['tick_width'], length=CONFIG['tick_length'])
 
    ax.legend(handles=legend_handles, labels=legend_labels, fontsize=12,
              loc='upper left', frameon=True, edgecolor='black')
    plt.tight_layout()
    
    # Save the figure (added as requested)
    plt.savefig("/path_to/neutron_attenuation_coeff.png", dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.savefig("/path_to/neutron_attenuation_coeff.svg", dpi='figure', format='svg', bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', transparent=True, backend='svg')

    #plt.show()

generate_attenuation_plot()
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


linestyle_tuple = [
    ('loosely dotted',        (0, (1, 10))),                # 0
    ('dotted',                (0, (1, 5))),                 # 1
    ('densely dotted',        (0, (1, 1))),                 # 2
    ('long dash with offset', (5, (10, 3))),                # 3
    ('loosely dashed',        (0, (5, 10))),                # 4
    ('dashed',                (0, (5, 5))),                 # 5
    ('densely dashed',        (0, (5, 1))),                 # 6
    ('loosely dashdotted',    (0, (3, 10, 1, 10))),         # 7
    ('dashdotted',            (0, (3, 5, 1, 5))),           # 8
    ('densely dashdotted',    (0, (3, 1, 1, 1))),           # 9
    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),     # 10
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),  # 11
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),     # 12
]


def get_y_range_zoom(t, x1, x2, y, t_offset=0):
    dt = t[1] - t[0]

    nx1 = int((x1-t_offset) / dt)
    nx2 = int((x2-t_offset) / dt)

    minimum = np.min(y[nx1:nx2+1])
    maximum = np.max(y[nx1:nx2+1])
    
    return 0.9 * minimum, 1.1 * maximum


def plot_trajectory(
    t,
    u_delay, u_delay_ml,
    control_delay, control_delay_ml,
    predictors, predictors_ml,
    savefig=None,
    axis=None,
):
    fig = plt.figure(figsize=set_size(469.75502, 1, (3, 2), height_add=0))
    gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1])  # 2 rows, 3 columns

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax6 = fig.add_subplot(gs[1, 4:6])
    ax7 = fig.add_subplot(gs[2, 0:3])
    ax8 = fig.add_subplot(gs[2, 3:6])
    

    style1 = {'color': 'tab:green', 'linestyle': linestyle_tuple[2][1], 'linewidth': 2}
    style2 = {'color': 'tab:orange', 'linestyle': linestyle_tuple[5][1], 'linewidth': 2, 'alpha': 0.7}


    ax1.plot(t, u_delay[:, 0], label="Const delay", **style1)
    ax1.plot(t, u_delay_ml[:, 0], label="Const delay ML", **style2)
    ax1.set_ylabel(r"$x(t)$")
    ax1.set_xlabel("t")
    # ax1.set_xticks([0, 2.5, 5, 7.5, 10])
    # ax1.set_yticks([1, 0.5, 0, -0.5, -1])

    ax2.plot(t, u_delay[:, 1], **style1)
    ax2.plot(t, u_delay_ml[:, 1], **style2)
    ax2.set_ylabel(r"$y(t)$", labelpad=2)
    ax2.set_xlabel("t")
    # ax2.set_xticks([0, 2.5, 5, 7.5, 10])
    # ax2.set_yticks([-0.5, 0, 0.5, 1])
    
    if axis: 
        # Create inset axes (zoom factor = 2)
        axins = inset_axes(ax2, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),  # (x0, y0, width, height)
                       bbox_transform=ax2.transAxes,
                       borderpad=0)  # Location of inset
        axins.plot(t, u_delay[:, 1], **style1)
        axins.plot(t, u_delay_ml[:, 1], **style2)
        

        # Limit the region shown in inset
        x1, x2 = 5.5, 6
        y1, y2 = get_y_range_zoom(t, x1, x2, u_delay[:, 1])
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # Hide tick labels for the inset
        axins.set_xticks([])
        axins.set_yticks([])

        # Draw lines connecting inset to main plot
        mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax3.plot(t, u_delay[:, 2], **style1)
    ax3.plot(t, u_delay_ml[:, 2], **style2)
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$\theta(t)$", labelpad=2)
    # ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    # ax3.set_xticks([0, 2.5, 5, 7.5, 10])

    # (N, m, NX, n)
    ax4.plot(t, predictors[:, 1, -1, 0], label="Const delay", **style1)
    ax4.plot(t, predictors_ml[:, 1, -1, 0], label="Const delay ML", **style2)
    ax4.set_ylabel(r"$P_{2_{X_1}}(t) \approx x(t+D(t))$")
    ax4.set_xlabel("t")
    # ax4.set_xticks([0, 2.5, 5, 7.5, 10])
    # ax4.set_yticks([1, 0.5, 0, -0.5, -1])

    # (N, m, NX, n)
    ax5.plot(t, predictors[:, 1, -1, 1], label="Const delay", **style1)
    ax5.plot(t, predictors_ml[:, 1, -1, 1], label="Const delay ML", **style2)
    ax5.set_ylabel(r"$P_{2_{X_2}}(t) \approx y(t+D(t))$", labelpad=2)
    ax5.set_xlabel("t")
    # ax5.set_xticks([0, 2.5, 5, 7.5, 10])
    # ax5.set_yticks([-0.5, 0, 0.5, 1])

    if axis:
        # Create inset axes (zoom factor = 2)
        axins2 = inset_axes(ax5, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),  # (x0, y0, width, height)
                       bbox_transform=ax5.transAxes,
                       borderpad=0)  # Location of inset
        axins2.plot(t, predictors[:, 1, -1, 1], label="Const delay", **style1)
        axins2.plot(t, predictors_ml[:, 1, -1, 1], label="Const delay ML", **style2)
        

        # Limit the region shown in inset
        x1, x2 = 5.5, 6
        y1, y2 = get_y_range_zoom(t, x1, x2, predictors[:, 1, -1, 1])
        axins2.set_xlim(x1, x2)
        axins2.set_ylim(y1, y2)

        # Hide tick labels for the inset
        axins2.set_xticks([])
        axins2.set_yticks([])
        mark_inset(ax5, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # (N, m, NX, n)
    ax6.plot(t, predictors[:, 1, -1, 2], label="Const delay", **style1)
    ax6.plot(t, predictors_ml[:, 1, -1, 2], label="Const delay ML", **style2)
    ax6.set_xlabel("t")
    ax6.set_ylabel(r"$P_{2_{X_3}}(t) \approx \theta(t+D(t))$", labelpad=2)
    ax6.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax6.set_xticks([0, 2.5, 5, 7.5, 10])
    plt.subplots_adjust(hspace=0.5, left=0.1, right=0.98, top=0.95, bottom=0.14, wspace=1)


    ax7.plot(t, control_delay[:, 0], **style1)
    ax7.plot(t, control_delay_ml[:, 0], **style2)
    ax7.set_xlabel("t")
    ax7.set_ylabel(r"$\nu_1(t)$")
    # ax7.set_yticks([-3, -1.5, 0, 1.5])
    # ax7.set_xticks([0, 2.5, 5, 7.5, 10])

    l1, = ax8.plot(t, control_delay[:, 1], label="Successive Approximations", **style1)
    l2, = ax8.plot(t, control_delay_ml[:, 1], label="ML", **style2)
    ax8.set_xlabel("t")
    ax8.set_ylabel(r"$\nu_2(t)$", labelpad=2)
    # ax8.set_yticks([-3, -2, -1,0,1])
    # ax8.set_xticks([0, 2.5, 5, 7.5, 10])


    fig.text(0.5, 0.98, "System states", va='center', ha='center', fontsize=16)
    fig.text(0.5, 0.67, "Predictions", va='center', ha='center', fontsize=16)
    fig.text(0.5, 0.37, "Control Inputs", va='center', ha='center', fontsize=16)


    # Add a horizontal legend below all subplots
    fig.legend(handles=[l1, l2],loc='lower center',ncol=3,fontsize=10, frameon=True,fancybox=True, shadow=False,bbox_to_anchor=(0.5, 0.02))
    if savefig is not None:
        abs_fig_path = (Path(__file__).parent.parent / f'media/{savefig}.png').resolve()
        plt.savefig(abs_fig_path, dpi=300)
    plt.show()


def plot_trajectory_without_predictors(
    t,
    u_delay, u_delay_deeponet, u_delay_fno,
    control_delay, control_delay_deeponet, control_delay_fno,
    savefig=None,
    axis=None,
):
    # Adjust figure size to be taller to accommodate titles
    fig = plt.figure(figsize=set_size(469.75502, 1, (2, 2), height_add=1.0))
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1], hspace=0.6, wspace=1)
    
    # First row (system states)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    
    # Second row (control inputs)
    ax7 = fig.add_subplot(gs[1, 0:3])
    ax8 = fig.add_subplot(gs[1, 3:6])
    
    style1 = {'color': (0, 0, 0), 'linestyle': linestyle_tuple[2][1], 'linewidth': 2}
    style2 = {'color': 'tab:red', 'linestyle': linestyle_tuple[5][1], 'linewidth': 2}
    style3 = {'color': 'tab:blue', 'linestyle': linestyle_tuple[8][1], 'linewidth': 2}

    # Plot system states
    ax1.plot(t, u_delay[:, 0], **style1)
    ax1.plot(t, u_delay_deeponet[:, 0], **style2)
    ax1.plot(t, u_delay_fno[:, 0], **style3)
    ax1.set_xlabel("t", labelpad=1)
    ax1.set_ylabel(r"$x(t)$", labelpad=2)

    ax2.plot(t, u_delay[:, 1], **style1)
    ax2.plot(t, u_delay_deeponet[:, 1], **style2)
    ax2.plot(t, u_delay_fno[:, 1], **style3)
    ax2.set_xlabel("t", labelpad=1)
    ax2.set_ylabel(r"$y(t)$", labelpad=2)
    
    if axis: 
        axins = inset_axes(ax2, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),
                       bbox_transform=ax2.transAxes, borderpad=0)
        axins.plot(t, u_delay[:, 1], **style1)
        axins.plot(t, u_delay_deeponet[:, 1], **style2)
        axins.plot(t, u_delay_fno[:, 1], **style3)
        
        x1, x2 = 5.5, 6
        y1, y2 = get_y_range_zoom(t, x1, x2, u_delay[:, 1])
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax3.plot(t, u_delay[:, 2], **style1)
    ax3.plot(t, u_delay_deeponet[:, 2], **style2)
    ax3.plot(t, u_delay_fno[:, 2], **style3)
    ax3.set_xlabel("t", labelpad=1)
    ax3.set_ylabel(r"$\theta(t)$", labelpad=2)

    # Plot control inputs
    ax7.plot(t, control_delay[:, 0], **style1)
    ax7.plot(t, control_delay_deeponet[:, 0], **style2)
    ax7.plot(t, control_delay_fno[:, 0], **style3)
    ax7.set_xlabel("t", labelpad=1)
    ax7.set_ylabel(r"$\nu_1(t)$", labelpad=2)
    ax7.set_yticks([-15, -10, -5, 0, 5])

    l1, = ax8.plot(t, control_delay[:, 1], label="Fixed point iteration", **style1)
    l2, = ax8.plot(t, control_delay_deeponet[:, 1], label="DeepONet", **style2)
    l3, = ax8.plot(t, control_delay_fno[:, 1], label="FNO", **style3)
    ax8.set_xlabel("t", labelpad=1)
    ax8.set_ylabel(r"$\nu_2(t)$", labelpad=2)
    ax8.set_yticks([-6, -4, -2, 0, 2])

    # Add section titles with more vertical spacing
    fig.text(0.5, 0.92, "System states", va='center', ha='center', fontsize=16)
    fig.text(0.5, 0.50, "Control Inputs", va='center', ha='center', fontsize=16)

    # Add legend with more bottom padding
    fig.legend(handles=[l1, l2, l3], loc='lower center', ncol=3, fontsize=10,
              frameon=True, fancybox=True, shadow=False,
              bbox_to_anchor=(0.5, 0.04))
    
    # Adjust subplot parameters to prevent overlap
    plt.subplots_adjust(
        hspace=0.6,
        left=0.1,
        right=0.98,
        top=0.88,
        bottom=0.18,
        wspace=1
    )
    
    if savefig is not None:
        abs_fig_path = (Path(__file__).parent.parent / f'media/{savefig}.png').resolve()
        plt.savefig(abs_fig_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_system_states_only(
    t,
    u_delay, u_delay_deeponet, u_delay_fno,
    savefig=None,
    axis=None,
):
    # Adjust figure size to be taller to accommodate titles
    w, h = set_size(469.75502, height_add=0.5)
    fig = plt.figure(figsize=(w,0.45*h))
    gs = gridspec.GridSpec(1, 3, height_ratios=[1], hspace=0.6)
    
    # System states plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    style1 = {'color': (0, 0, 0), 'linestyle': linestyle_tuple[2][1], 'linewidth': 2}
    style2 = {'color': 'tab:red', 'linestyle': linestyle_tuple[5][1], 'linewidth': 2}
    style3 = {'color': 'tab:blue', 'linestyle': linestyle_tuple[8][1], 'linewidth': 2}

    # Plot system states
    ax1.plot(t, u_delay[:, 0], **style1)
    ax1.plot(t, u_delay_deeponet[:, 0], **style2)
    ax1.plot(t, u_delay_fno[:, 0], **style3)
    ax1.set_xlabel("t", labelpad=1)
    ax1.set_ylabel(r"$x(t)$", labelpad=2)

    ax2.plot(t, u_delay[:, 1], **style1)
    ax2.plot(t, u_delay_deeponet[:, 1], **style2)
    ax2.plot(t, u_delay_fno[:, 1], **style3)
    ax2.set_xlabel("t", labelpad=1)
    ax2.set_ylabel(r"$y(t)$", labelpad=2)
    
    if axis: 
        axins = inset_axes(ax2, width="30%", height="30%", bbox_to_anchor=(-0.2, -0.2, 1,1),
                       bbox_transform=ax2.transAxes, borderpad=0)
        axins.plot(t, u_delay[:, 1], **style1)
        axins.plot(t, u_delay_deeponet[:, 1], **style2)
        axins.plot(t, u_delay_fno[:, 1], **style3)
        
        x1, x2 = 5.5, 6
        y1, y2 = get_y_range_zoom(t, x1, x2, u_delay[:, 1])
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax3.plot(t, u_delay[:, 2], **style1)
    ax3.plot(t, u_delay_deeponet[:, 2], **style2)
    ax3.plot(t, u_delay_fno[:, 2], **style3)
    ax3.set_xlabel("t", labelpad=1)
    ax3.set_ylabel(r"$\theta(t)$", labelpad=2)

    # Add section title
    fig.text(0.5, 0.92, "System states", va='center', ha='center', fontsize=14)

    # Add legend with more bottom padding
    l1, = ax3.plot([], [], label="Fixed point iteration", **style1)
    l2, = ax3.plot([], [], label="DeepONet", **style2)
    l3, = ax3.plot([], [], label="FNO", **style3)
    
    fig.legend(handles=[l1, l2, l3], loc='lower center', ncol=3, fontsize=7,
              frameon=True, fancybox=True, shadow=False,
              bbox_to_anchor=(0.5, 0.00))
    
    # Adjust subplot parameters to prevent overlap
    plt.subplots_adjust(
        hspace=0.3,
        left=0.1,
        right=0.98,
        top=0.86,
        bottom=0.27,
        wspace=0.5
    )
    
    if savefig is not None:
        abs_fig_path = (Path(__file__).parent.parent / f'media/{savefig}.png').resolve()
        plt.savefig(abs_fig_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/bel fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    plt.rcParams.update(tex_fonts)

    T, dt = 1, 0.01
    t = np.arange(0, T + dt, dt)
    n = len(t)
    plot_system_states_only(
        t,
        np.zeros((n,3)),np.zeros((n,3)),np.zeros((n,3)),
        savefig='single_trajectory_test'
    )
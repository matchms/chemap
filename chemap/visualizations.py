import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def heatmap_comparison(similarities1, similarities2, label1, label2, bins=50,
                       colormap="inferno", ignore_diagonal=True, 
                       add_region_percentage=True, filename=None,
                       dpi=300,
                      ):
    """
    Generates a heatmap comparison of two similarity matrices.
    
    Parameters:
    similarities1 (ndarray): First similarity matrix (NxN) with values in a comparable range.
    similarities2 (ndarray): Second similarity matrix (NxN) with values in a comparable range.
    label1 (str): Label for the x-axis.
    label2 (str): Label for the y-axis.
    bins (int, optional): Number of bins for the 2D histogram (default: 50).
    colormap (str, optional): Colormap used for visualization (default: "viridis").
    ignore_diagonal (bool, optional): If True, excludes the diagonal values (default: True).
    filename (str, optional): If provided, saves the heatmap to the specified file.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Select upper triangle indices to ignore diagonal if necessary
    n = similarities1.shape[0]
    mask_indices = np.triu_indices(n, k=1 if ignore_diagonal else 0)
    
    # Compute the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        similarities1[mask_indices], 
        similarities2[mask_indices], 
        bins=bins
    )
    
    # Plot the heatmap using imshow with a logarithmic color scale
    im = ax.imshow(
        hist.T, origin='lower', aspect='equal',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=colormap, norm=LogNorm(vmin=1, vmax=np.max(hist))
    )
    
    # Create an axis of the same height for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Count")
    
    # Compute total count for percentage calculations
    total_count = len(mask_indices[0])
    
    # Add text annotations for each 5x5 section of bins
    step = bins // 5
    for i in range(0, bins, step):
        for j in range(0, bins, step):
            sub_matrix = hist[i:i+step, j:j+step]
            subsection_sum = np.sum(sub_matrix)
            
            if subsection_sum > 0 and add_region_percentage:
                # Compute the center of the bin for text placement
                x_center = (x_edges[i] + x_edges[min(i + step, bins - 1)]) / 2
                y_center = (y_edges[j] + y_edges[min(j + step, bins - 1)]) / 2
                
                # Add text annotation
                txt = ax.text(
                    x_center, y_center,
                    f"{(100 * subsection_sum / total_count):.2f}%",
                    color="white",
                    ha="center", va="center",
                    fontsize=8, zorder=2
                )

                # Add black outline
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1.5, foreground="black"),
                    path_effects.Normal()
                ])
    
    # Configure grid, labels, and layout
    ax.grid(True, zorder=1)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    plt.tight_layout()
    
    # Save plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=dpi)
    
    # Show the plot
    plt.show()


###############################################################################
# 1) A helper to create non-uniform bin edges (finer near 100%)
###############################################################################
def make_scaled_bin_edges(num_bins=50, epsilon=0.01):
    """
    Return a non-uniform array of bin edges in [0,100],
    which is denser near the top (100%).
    """
    def forward_transform(p):
        # p in [0,100] -> log-like transform
        p_clipped = np.clip(p, 0, 100) / 100.0
        return -np.log((1.0 + epsilon) - p_clipped)
    
    def inverse_transform(x):
        # x -> p in [0,100]
        return 100.0 * ((1.0 + epsilon) - np.exp(-x))

    t_min = forward_transform(0)     # -log(1+epsilon)
    t_max = forward_transform(100)   # -log(epsilon)

    t_values = np.linspace(t_min, t_max, num_bins + 1)
    p_edges = inverse_transform(t_values)
    return p_edges

###############################################################################
# 2) A helper to convert a desired percentile p in [0,100] to the uniform
#    coordinate [0..1], given the non-uniform bin edges.
###############################################################################
def percentile_to_uniform(p, edges):
    """
    Given a desired percentile p in [0,100], find its 'uniform coordinate' 
    in [0,1] based on the non-uniform bin edges array (length = bins+1).

    - edges[i], edges[i+1] define the percentile range for bin i.
    - If p is in [edges[i], edges[i+1]), we place it proportionally
      within bin i in [0..1].
    """
    # Find the bin index where p belongs
    i = np.searchsorted(edges, p, side='right') - 1
    # Clamp i to [0, len(edges)-2]
    i = max(0, min(i, len(edges) - 2))
    
    # Fractional offset within that bin
    bin_left = edges[i]
    bin_right = edges[i+1]
    width = max(bin_right - bin_left, 1e-12)
    frac = (p - bin_left) / width
    
    # Normalize to [0..1] across all bins
    x_uni = (i + frac) / (len(edges) - 1)
    return x_uni

###############################################################################
# 3) The main function
###############################################################################
def heatmap_comparison_scaled_bins(similarities1, similarities2, 
                                   label1, label2,
                                   bins=50, colormap="inferno",
                                   ignore_diagonal=True, epsilon=0.01,
                                   add_region_percentage=True,
                                   filename=None):
    """
    similarities1, similarities2: 2D arrays of shape (N, N) in [0,100] (percentiles).
    We'll:
      - Bin with non-uniform edges (finer near 100%).
      - Plot via imshow in uniform 0..1 space (each bin = same size).
      - Then define major percentile intervals, sum those big boxes, and label them.
      - Add major ticks (labeled) and minor ticks (unlabeled) accordingly.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # -------------------------------------------------------------------------
    # 1) Data selection
    # -------------------------------------------------------------------------
    n = similarities1.shape[0]
    if ignore_diagonal:
        iu1 = np.triu_indices(n, k=1)
    else:
        iu1 = np.triu_indices(n, k=0)
    
    # -------------------------------------------------------------------------
    # 2) Non-uniform bin edges + histogram
    # -------------------------------------------------------------------------
    x_edges = make_scaled_bin_edges(num_bins=bins, epsilon=epsilon)  # length = bins+1
    y_edges = make_scaled_bin_edges(num_bins=bins, epsilon=epsilon)  # length = bins+1
    
    hist, _, _ = np.histogram2d(
        similarities1[iu1],
        similarities2[iu1],
        bins=[x_edges, y_edges]
    )
    
    # -------------------------------------------------------------------------
    # 3) Plot the heatmap in [0..1] x [0..1]
    # -------------------------------------------------------------------------
    im = ax.imshow(
        hist.T,
        origin='lower',
        aspect='equal',
        extent=[0, 1, 0, 1],
        cmap=colormap,
        norm=LogNorm(vmin=1, vmax=hist.max() if hist.max() > 0 else 1)
    )
    
    # Create an axis of the same height for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding as needed
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Count")

    # -------------------------------------------------------------------------
    # 4) Define major intervals & label each box's fraction
    #    Example: [0, 50, 80, 95, 99, 100]
    # -------------------------------------------------------------------------
    major_percentiles = [0, 60, 85, 95, 99, 100]
    total_sum = hist.sum()
    
    # Convert major cutoffs to uniform space
    x_major_coords = [percentile_to_uniform(p, x_edges) for p in major_percentiles]
    y_major_coords = [percentile_to_uniform(p, y_edges) for p in major_percentiles]
      
    def coord_to_bin_idx(u):
        """Convert a uniform coordinate in [0..1] to a histogram bin index [0..bins)."""
        # e.g. u=0.3 => bin 15 if bins=50
        idx = int(np.floor(u * bins))
        # clamp
        return max(0, min(idx, bins - 1))

    for ix in range(len(major_percentiles) - 1):
        x_low = x_major_coords[ix]
        x_high = x_major_coords[ix + 1]
        
        x_lo_bin = major_percentiles[ix]
        x_hi_bin = major_percentiles[ix + 1]
        if x_lo_bin == 0:
            x_lo_bin = - 0.00001
        
        for iy in range(len(major_percentiles) - 1):
            y_low = y_major_coords[iy]
            y_high = y_major_coords[iy + 1]

            y_lo_bin = major_percentiles[iy]
            y_hi_bin = major_percentiles[iy + 1]
            if y_lo_bin == 0:
                y_lo_bin = - 0.00001
            sub_matrix = np.where(
                (similarities1[iu1] > x_lo_bin) & (similarities1[iu1] <= x_hi_bin) \
                & (similarities2[iu1] > y_lo_bin) & (similarities2[iu1] <= y_hi_bin)
            )[0]
            sub_sum = sub_matrix.shape[0]
            if sub_sum > 0 and add_region_percentage:
                fraction = 100.0 * sub_sum / total_sum
                # Place text at the center of the big box
                x_center = 0.5 * (x_low + x_high)
                y_center = 0.5 * (y_low + y_high)
                
                txt = ax.text(
                    x_center, y_center,
                    f"{fraction:.2f}%", color="white",
                    ha="center", va="center", fontsize=8
                )

                # Add black outline
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1.5, foreground="black"),
                    path_effects.Normal()
                ])
    # -------------------------------------------------------------------------
    # 5) Axis Ticks: major vs. minor
    #    For example:
    #      major: [0, 50, 80, 95, 99, 100]
    #      minor: [0,10,20,30,40,50,60,70,80,90,95,99,100]
    # -------------------------------------------------------------------------
    minor_percentiles = [0,10,20,30,40,50,60,70,80,85,90,95,97,99,99.7,100]
    
    # Convert to uniform space
    x_major_pos = [percentile_to_uniform(p, x_edges) for p in major_percentiles]
    y_major_pos = [percentile_to_uniform(p, y_edges) for p in major_percentiles]
    x_minor_pos = [percentile_to_uniform(p, x_edges) for p in minor_percentiles]
    y_minor_pos = [percentile_to_uniform(p, y_edges) for p in minor_percentiles]
    
    # Set major ticks
    ax.set_xticks(x_major_pos, minor=False)
    ax.set_yticks(y_major_pos, minor=False)
    ax.set_xticklabels([f"{p}%" for p in major_percentiles], minor=False, rotation=90)
    ax.set_yticklabels([f"{p}%" for p in major_percentiles], minor=False)
    
    # Set minor ticks
    ax.set_xticks(x_minor_pos, minor=True)
    ax.set_yticks(y_minor_pos, minor=True)
    ax.set_xticklabels([f"{p}%" for p in minor_percentiles], minor=True, rotation=90)
    ax.set_yticklabels([f"{p}%" for p in minor_percentiles], minor=True)
    
    # Optionally turn on grid lines
    ax.grid(which='major', color='lightgray', linestyle='-', linewidth=0.8, alpha=0.9)
    #ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

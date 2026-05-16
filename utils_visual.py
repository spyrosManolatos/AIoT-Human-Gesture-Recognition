"""
This module contains functions for visualizing the movement dataset and the results of the processing and feature engineering steps. It includes functions for
plotting the count of segmented instances per gesture and user, comparing raw and filtered signals, visualizing movement instances in time domain and
3D, plotting heatmaps, pairwise scatter plots for feature correlation verification, and boxplots for feature distributions across classes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
def plot_window_counts_by_gesture_user(
        windows:list
):
    """Plots the count of segmented instances per gesture and user.

    Args:
        windows: A list of the segmented instances (windows) of the movement
            dataset.
    Returns:
    """
    seg_counts = pd.DataFrame([
        {
            "gesture_id": w["gesture_id"].iloc[0],
            "user": w["user"].iloc[0]
        }
        for w in windows
    ])
    bar_df = (
        seg_counts.groupby(["gesture_id", "user"])
        .size()
        .reset_index(name="window_count")
    )
    bar_df["gesture_user"] = (
        bar_df["gesture_id"].astype(str) + "_" + bar_df["user"].astype(str)
    )
    plt.figure(figsize=(14, 7))
    sns.barplot(data=bar_df, x="gesture_user", y="window_count", color="steelblue")

    plt.title("Count of Windows per Gesture and User")
    plt.xlabel("Gesture ID and User")
    plt.ylabel("Number of Windows")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_filtered_vs_raw_signal(
        raw_signals: list,
        filtered_signals: list,
        idx = 0,
        n_start: int = 0,
        n_end: int = 100,
):
    """Plots the raw and the filtered signal of a movement instance.

    Args:
        raw_signals: A list of the raw signals of the movement instances.
        filtered_signals: A list of the filtered signals of the movement
            instances.
        idx: The index of the signal to be plotted. For example, if idx = 0, the first signal (swipe up of user 01) will be plotted. If idx = 1, the second signal (swipe down of user 1) will be plotted, etc.
        n_start: The starting index of the samples to be plotted.
        n_end: The ending index of the samples to be plotted.
    Returns:
    """

    filtered_instance = filtered_signals[idx].iloc[n_start:n_end].reset_index(drop=True)
    non_filtered_instance = raw_signals[idx].iloc[n_start:n_end].reset_index(drop=True)

    signal_cols = [col for col in filtered_instance.columns if col not in ["gesture_id", "user"]]

    fig, axes = plt.subplots(len(signal_cols), 1, figsize=(12, 14), sharex=True)
    fig.suptitle("Comparison of Filtered and Non-Filtered Signals", fontsize=14)

    for ax, column in zip(axes, signal_cols):
        ax.plot(non_filtered_instance[column], label=f"Non-Filtered {column}", alpha=0.5)
        ax.plot(filtered_instance[column], label=f"Filtered {column}", linestyle="--")
        ax.set_ylabel(column)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

def plot_instance_time_domain(df: pd.DataFrame):
    """Visualizes the movement instance to a plot in time domain.

    Args:
        df: The DataFrame to be visualized in time domain.

    Returns:

    """
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20).legend(fontsize=18)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_instance_3d(
        df: pd.DataFrame,
        axes_list: tuple = ("acc_x", "acc_y", "acc_z")
):
    """Plots a 3-axes DataFrame in 3D graph.

    Args:
        df: The DataFrame to be plotted in 3D.
        axes_list: Tuple with the 3-axis values. For gyroscope axes should
            be: ("gyr_x", "gyr_y", "gyr_z")

    Returns:

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # print the plot in 3D

    xs = df[axes_list[0]]
    ys = df[axes_list[1]]
    zs = df[axes_list[2]]

    ax.scatter(xs, ys, zs, color='green', s=50, alpha=0.6, edgecolors='w')

    ax.set_xlabel(axes_list[0])
    ax.set_ylabel(axes_list[1])
    ax.set_zlabel(axes_list[2])


def plot_np_instance(
        np_array: np.ndarray,
        columns_list: list
):
    """Plot NumPy instance using DataFrames (pandas). It first transforms the
        array into
    DataFrame with its corresponding columns naming, and then, it plots the
        DataFrame in time domain.

    Args:
        np_array: The NumPy array to be transformed.
        columns_list: The columns list that the DataFrame and the plot will
            have.

    Returns:

    """
    df = pd.DataFrame(np_array, columns=columns_list)
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_heatmap(df: pd.DataFrame):
    """Visualizes the heatmap of the DataFrame's values.

    Args:
        df: A DataFrame.

    Returns:

    """
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap='plasma')

def plot_pairwise_scatter(
        df: pd.DataFrame,
        features: list,
        label_col: str = "gesture_id",
        display_type: str = "grid",
        output_dir: str = None,
):
    """Pairwise scatter plots to verify feature correlations across multiple classes.

    Args:
        df: DataFrame containing features + label column.
        features: List of feature column names to plot.
        label_col: Column name for the class/gesture label.
        display_type: "grid" (default) displays plots in matplotlib grid.
                      "files" saves individual PNG files to output_dir.
        output_dir: Directory to save plots (only used if display_type="files").

    Returns:
        None (displays plots or saves to files)
    """
    import itertools
    import os
    from itertools import combinations

    plot_df = df[features + [label_col]].dropna()
    
    pairs = list(combinations(features, 2))
    n_pairs = len(pairs)
    
    if display_type == "grid":
        # Display all plots in a grid
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_pairs == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        
        fig.suptitle("Feature Correlation Verification: Pairwise Scatter Plots", 
                     fontsize=16, fontweight="bold", y=0.995)
        
        colors = sns.color_palette("Set2", n_colors=len(plot_df[label_col].unique()))
        
        for idx, (feat_a, feat_b) in enumerate(pairs):
            ax = axes[idx] if n_pairs > 1 else axes[0]
            
            # Scatter plot colored by class
            for i, cls in enumerate(plot_df[label_col].unique()):
                mask = plot_df[label_col] == cls
                ax.scatter(plot_df.loc[mask, feat_a],
                          plot_df.loc[mask, feat_b],
                          c=[colors[i]], s=30, alpha=0.6, label=cls, edgecolors='w', linewidth=0.5)
            
            # Calculate correlation
            r = np.corrcoef(plot_df[feat_a], plot_df[feat_b])[0, 1]
            
            # Color code based on correlation strength
            r_col = ("#E74C3C" if abs(r) > 0.85
                     else "#F39C12" if abs(r) > 0.60
                     else "#27AE60")
            
            ax.set_xlabel(feat_a, fontsize=10)
            ax.set_ylabel(feat_b, fontsize=10)
            ax.set_title(f"r = {r:.3f}", fontsize=11, fontweight="bold", color=r_col)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=8, loc="best", ncol=2)
        
        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        print(f"Displayed {n_pairs} pairwise scatter plots in grid format.")
        print(f"Red (r > 0.85) = Redundant | Orange (r > 0.60) = Moderately correlated | Green (r ≤ 0.60) = Independent")

    elif display_type == "files":
        # Save individual plots to files (original behavior)
        if output_dir is None:
            output_dir = "pair_plots"
        
        palette = {cls: colors[i] for i, cls in enumerate(plot_df[label_col].unique())}
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {n_pairs} pair plots to '{output_dir}/' ...")
        
        for feat_a, feat_b in pairs:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Scatter plot colored by class
            for cls, col in palette.items():
                mask = plot_df[label_col] == cls
                ax.scatter(plot_df.loc[mask, feat_a],
                          plot_df.loc[mask, feat_b],
                          c=[col], s=30, alpha=0.6, label=cls, edgecolors='w', linewidth=0.5)
            
            # Calculate correlation
            r = np.corrcoef(plot_df[feat_a], plot_df[feat_b])[0, 1]
            
            # Color code based on correlation strength
            r_col = ("#E74C3C" if abs(r) > 0.85
                     else "#F39C12" if abs(r) > 0.60
                     else "#27AE60")
            
            status = ("⚠ REDUNDANT" if abs(r) > 0.85
                      else "~ moderately correlated" if abs(r) > 0.60
                      else "✓ independent")
            
            ax.set_xlabel(feat_a, fontsize=11)
            ax.set_ylabel(feat_b, fontsize=11)
            ax.set_title(f"{feat_a} vs {feat_b}  |  r = {r:.3f}  [{status}]", 
                        fontsize=12, fontweight="bold", color=r_col)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fname = f"{feat_a}__{feat_b}.png".replace("/", "-")
            fig.savefig(os.path.join(output_dir, fname), dpi=110, bbox_inches="tight")
            plt.close(fig)
        
        print(f"Saved {n_pairs} plots. Done.")


def plot_feature_boxplots_by_class(
        df: pd.DataFrame,
        features: list,
        label_col: str = "gesture_id",
        figsize_per_plot: tuple = (6, 5)
):
    """Create boxplots for multiple features, each showing distributions across gesture classes.
    
    Each subplot shows one feature with boxplots for each gesture class on the x-axis.
    
    Args:
        df: DataFrame containing features and label column
        features: List of feature column names to plot
        label_col: Column name for gesture/activity labels
        figsize_per_plot: Size per subplot (width, height)
    
    Returns:
        None (displays plots)
    """
    n_features = len(features)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_features + n_cols - 1) // n_cols
    
    figsize = (n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes if needed
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    fig.suptitle(f"Feature Box Plots by {label_col.replace('_', ' ').title()}", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Prepare data for boxplot
        data_for_plot = df[[feature, label_col]].copy()
        
        # Create boxplot
        sns.boxplot(
            data=data_for_plot,
            x=label_col,
            y=feature,
            ax=ax,
            palette="Set2",
            width=0.6
        )
        
        # Formatting
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_xlabel(label_col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        
        # Rotate x labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed boxplots for {n_features} features grouped by {label_col}")


def plot_scatter_pca(
        df: pd.DataFrame,
        c_name: str,
        cmap_set: str = "plasma"
):
    """Visualizes the values of the component columns of the DataFrame
    according to its column that includes the labels.

    Args:
        df: The DataFrame that contains the transformed data after the PCA
            procedure.
        c_name: The name of the column that includes the labels.
        cmap_set: The format of the plot.

    Returns:

    """
    if len(df.columns) == 3:
        plt.style.use('classic')
        plt.figure(figsize=(16, 8))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df[c_name], cmap=cmap_set)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
    elif len(df.columns) == 4:
        plt.style.use('classic')
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[c_name], cmap=cmap_set)
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
    else:
        print("The DataFrame has more than 4 columns.")
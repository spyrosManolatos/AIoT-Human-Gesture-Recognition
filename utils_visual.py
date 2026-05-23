"""
This module contains functions for visualizing the movement dataset and the results of the processing and feature engineering steps. It includes functions for
plotting the count of segmented instances per gesture and user, comparing raw and filtered signals, visualizing movement instances in time domain and
3D, plotting heatmaps, pairwise scatter plots for feature correlation verification, and boxplots for feature distributions across classes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# ==========================================
# 1. Exploratory Data Analysis (EDA)
# ==========================================

def plot_window_counts_by_gesture_user(
        windows: list,
        save_fig: bool = True,
        output_dir: str = "figures/eda"
):
    """Plots the count of segmented instances per gesture and user.

    Args:
        windows: A list of the segmented instances (windows) of the movement dataset.
        save_fig: Whether to save the plot to a file.
        output_dir: Directory to save the plot.
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
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "window_counts_by_gesture_user.png"), bbox_inches="tight")
    plt.show()


def plot_gesture_duration_by_user(
        df: pd.DataFrame,
        sampling_rate: int = 100,
        save_fig: bool = True,
        output_dir: str = "figures/eda",
        filename: str = "gesture_duration_by_user.png"
):
    """Plots the total duration (in seconds) of recordings for each gesture and user.

    Args:
        df: The DataFrame containing 'gesture_id' and 'user' columns.
        sampling_rate: Sensor sampling rate in Hz (default 100Hz).
        save_fig: Whether to save the figure to disk.
        output_dir: Directory to save the figure.
        filename: Filename for the saved plot.
    """
    duration_dict = {}
    for key, value in df.groupby(["gesture_id", "user"]):
        new_key = f"{key[0]}_{key[1]}"
        duration_dict[new_key] = len(value) / sampling_rate
        
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(duration_dict.keys()), y=list(duration_dict.values()), color="steelblue")
    plt.xticks(rotation=90)
    plt.xlabel("Gesture ID and User", fontsize=11)
    plt.ylabel("Duration (in seconds)", fontsize=11)
    plt.title("Total Duration of Each Gesture and User", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()


def plot_filtered_vs_raw_signal(
        raw_signals: list,
        filtered_signals: list,
        idx = 0,
        n_start: int = 0,
        n_end: int = 100,
        save_fig: bool = True,
        output_dir: str = "figures/eda"
):
    """Plots the raw and the filtered signal of a movement instance.

    Args:
        raw_signals: A list of the raw signals of the movement instances.
        filtered_signals: A list of the filtered signals of the movement instances.
        idx: The index of the signal to be plotted.
        n_start: The starting index of the samples to be plotted.
        n_end: The ending index of the samples to be plotted.
        save_fig: Whether to save the plot.
        output_dir: Directory to save the plot.
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
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"filtered_vs_raw_signal_idx_{idx}.png"), bbox_inches="tight")
    plt.show()


def plot_instance_time_domain(
        df: pd.DataFrame,
        save_fig: bool = True,
        output_dir: str = "figures/eda",
        filename: str = "instance_time_domain.png"
):
    """Visualizes the movement instance to a plot in time domain."""
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20).legend(fontsize=18)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()


def plot_instance_3d(
        df: pd.DataFrame,
        axes_list: tuple = ("acc_x", "acc_y", "acc_z"),
        save_fig: bool = True,
        output_dir: str = "figures/eda",
        filename: str = "instance_3d.png"
):
    """Plots a 3-axes DataFrame in 3D graph."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = df[axes_list[0]]
    ys = df[axes_list[1]]
    zs = df[axes_list[2]]

    ax.scatter(xs, ys, zs, color='green', s=50, alpha=0.6, edgecolors='w')

    ax.set_xlabel(axes_list[0])
    ax.set_ylabel(axes_list[1])
    ax.set_zlabel(axes_list[2])
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()


def plot_np_instance(
        np_array: np.ndarray,
        columns_list: list,
        save_fig: bool = True,
        output_dir: str = "figures/eda",
        filename: str = "np_instance.png"
):
    """Plot NumPy instance using DataFrames (pandas)."""
    df = pd.DataFrame(np_array, columns=columns_list)
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()


def plot_heatmap(
        df: pd.DataFrame,
        save_fig: bool = True,
        output_dir: str = "figures/eda",
        filename: str = "heatmap.png"
):
    """Visualizes the heatmap of the DataFrame's values."""
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap='plasma')
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()


# ==========================================
# 2. Feature Analysis & Selection
# ==========================================

def plot_pairwise_scatter(
        df: pd.DataFrame,
        features: list,
        label_col: str = "gesture_id",
        display_type: str = "grid",
        output_dir: str = "figures/features",
        save_fig: bool = True,
        filename: str = "pairwise_scatter_grid.png"
):
    """Pairwise scatter plots to verify feature correlations across multiple classes."""
    import itertools
    from itertools import combinations

    plot_df = df[features + [label_col]].dropna()
    pairs = list(combinations(features, 2))
    n_pairs = len(pairs)
    
    if display_type == "grid":
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
            
            for i, cls in enumerate(plot_df[label_col].unique()):
                mask = plot_df[label_col] == cls
                ax.scatter(plot_df.loc[mask, feat_a],
                          plot_df.loc[mask, feat_b],
                          c=[colors[i]], s=30, alpha=0.6, label=cls, edgecolors='w', linewidth=0.5)
            
            r = np.corrcoef(plot_df[feat_a], plot_df[feat_b])[0, 1]
            r_col = ("#E74C3C" if abs(r) > 0.85
                     else "#F39C12" if abs(r) > 0.60
                     else "#27AE60")
            
            ax.set_xlabel(feat_a, fontsize=10)
            ax.set_ylabel(feat_b, fontsize=10)
            ax.set_title(f"r = {r:.3f}", fontsize=11, fontweight="bold", color=r_col)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=8, loc="best", ncol=2)
        
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.show()
        
        print(f"Displayed {n_pairs} pairwise scatter plots in grid format.")
        print(f"Red (r > 0.85) = Redundant | Orange (r > 0.60) = Moderately correlated | Green (r ≤ 0.60) = Independent")

    elif display_type == "files":
        if output_dir == "figures/features":
            output_dir = "figures/features/pair_plots"
        elif output_dir is None:
            output_dir = "pair_plots"
        
        colors = sns.color_palette("Set2", n_colors=len(plot_df[label_col].unique()))
        palette = {cls: colors[i] for i, cls in enumerate(plot_df[label_col].unique())}
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {n_pairs} pair plots to '{output_dir}/' ...")
        
        for feat_a, feat_b in pairs:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            for cls, col in palette.items():
                mask = plot_df[label_col] == cls
                ax.scatter(plot_df.loc[mask, feat_a],
                          plot_df.loc[mask, feat_b],
                          c=[col], s=30, alpha=0.6, label=cls, edgecolors='w', linewidth=0.5)
            
            r = np.corrcoef(plot_df[feat_a], plot_df[feat_b])[0, 1]
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
        user: str = "all",
        figsize_per_plot: tuple = (6, 5),
        save_fig: bool = True,
        output_dir: str = "figures/features",
        filename: str = "feature_boxplots_by_class.png"
):
    """Create boxplots for multiple features, each showing distributions across gesture classes."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    figsize = (n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    fig.suptitle(f"Feature Box Plots by {label_col.replace('_', ' ').title()}{f' for User {user}' if user != 'all' else ''}", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        if user == "all":
            data_for_plot = df[[feature, label_col]].copy()
        else:
            data_for_plot = df[df["user"] == user][[feature, label_col]].copy()
        
        sns.boxplot(
            data=data_for_plot,
            x=label_col,
            y=feature,
            ax=ax,
            palette="Set2",
            width=0.6
        )
        
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_xlabel(label_col.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()
    
    print(f"Displayed boxplots for {n_features} features grouped by {label_col}")


def plot_anova_feature_comparison(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    k: int = 12,
    label_col: str = "gesture_id",
    user_col: str = "user",
    figsize: tuple = (14, 6),
    save_fig: bool = True,
    output_dir: str = "figures/subject_insights"
):
    """Run ANOVA F-test separately on train and test dataframes, compare and plot the top k features."""
    from sklearn.feature_selection import SelectKBest, f_classif
    
    metadata_cols = [label_col, user_col]
    X_train = df_train.drop(columns=[col for col in metadata_cols if col in df_train.columns]).dropna()
    y_train = df_train.loc[X_train.index, label_col]
    
    X_test = df_test.drop(columns=[col for col in metadata_cols if col in df_test.columns]).dropna()
    y_test = df_test.loc[X_test.index, label_col]
    
    selector_train = SelectKBest(f_classif, k=k)
    selector_train.fit(X_train, y_train)
    scores_train = pd.Series(selector_train.scores_, index=X_train.columns).sort_values(ascending=False)
    top_train = scores_train.head(k)
    
    selector_test = SelectKBest(f_classif, k=k)
    selector_test.fit(X_test, y_test)
    scores_test = pd.Series(selector_test.scores_, index=X_test.columns).sort_values(ascending=False)
    top_test = scores_test.head(k)
    
    intersection = list(set(top_train.index).intersection(set(top_test.index)))
    print("=" * 60)
    print(f"ANOVA Feature Selection Comparison (Top {k} Features)")
    print("=" * 60)
    print(f"Top {k} Features for Train (Users 01 & 02):")
    print(list(top_train.index))
    print(f"\nTop {k} Features for Test (User 03):")
    print(list(top_test.index))
    print(f"\nCommon Features ({len(intersection)} / {k}):")
    print(intersection if intersection else "None")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    sns.barplot(
        x=top_train.values,
        y=top_train.index,
        ax=axes[0],
        palette="viridis",
        hue=top_train.index,
        legend=False
    )
    axes[0].set_title(f"Top {k} ANOVA Features (Train: Users 01 & 02)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("F-Score")
    axes[0].grid(True, alpha=0.3)
    
    sns.barplot(
        x=top_test.values,
        y=top_test.index,
        ax=axes[1],
        palette="magma",
        hue=top_test.index,
        legend=False
    )
    axes[1].set_title(f"Top {k} ANOVA Features (Test: User 03)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("F-Score")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "anova_feature_comparison.png"), bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_fig: bool = True,
    output_dir: str = "figures/features",
    filename: str = "correlation_matrix.png"
):
    """Plots the correlation matrix of the DataFrame's values."""
    plt.figure(figsize=(10, 8))
    plt.title("Correlation Matrix", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. PCA & Dimensionality Reduction
# ==========================================

def plot_scatter_pca(
        df: pd.DataFrame,
        c_name: str,
        cmap_set: str = "plasma",
        save_fig: bool = True,
        output_dir: str = "figures/pca",
        filename: str = "scatter_pca.png"
):
    """Visualizes the values of the component columns of the DataFrame according to its label column."""
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
        return

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()


def plot_pca_train_test_comparison(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        features: list,
        label_col: str = "gesture_id",
        user_col: str = "user",
        figsize: tuple = (14, 6),
        save_fig: bool = True,
        output_dir: str = "figures/pca"
):
    """Plot PCA projection for both train and test sets to inspect domain shift."""
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    
    df_train_clean = df_train.dropna(subset=features).copy()
    df_test_clean = df_test.dropna(subset=features).copy()
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(df_train_clean[features])
    X_test_scaled = scaler.transform(df_test_clean[features])
    
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    train_plot_df = pd.DataFrame(X_train_pca, columns=["PC1", "PC2"])
    train_plot_df[label_col] = df_train_clean[label_col].values
    train_plot_df[user_col] = df_train_clean[user_col].values
    train_plot_df["Split"] = "Train"
    
    test_plot_df = pd.DataFrame(X_test_pca, columns=["PC1", "PC2"])
    test_plot_df[label_col] = df_test_clean[label_col].values
    test_plot_df[user_col] = df_test_clean[user_col].values
    test_plot_df["Split"] = "Test"
    
    combined_df = pd.concat([train_plot_df, test_plot_df], ignore_index=True)
    
    fig1, axes1 = plt.subplots(1, 2, figsize=figsize)
    
    sns.scatterplot(
        data=combined_df,
        x="PC1",
        y="PC2",
        hue="Split",
        style=label_col,
        markers=True,
        alpha=0.7,
        s=60,
        ax=axes1[0],
        palette="Set1"
    )
    axes1[0].set_title("Global PCA: Train vs Test Split", fontsize=12, fontweight="bold")
    axes1[0].grid(True, alpha=0.3)
    
    sns.scatterplot(
        data=combined_df,
        x="PC1",
        y="PC2",
        hue=user_col,
        style=label_col,
        markers=True,
        alpha=0.7,
        s=60,
        ax=axes1[1],
        palette="tab10"
    )
    axes1[1].set_title("Global PCA: Colored by User ID", fontsize=12, fontweight="bold")
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, "pca_train_test_global.png"), bbox_inches="tight")
    plt.show()
    
    gestures = sorted(combined_df[label_col].unique())
    num_gestures = len(gestures)
    
    fig2, axes2 = plt.subplots(1, num_gestures, figsize=(4.5 * num_gestures, 4.5), squeeze=False)
    
    for i, gest in enumerate(gestures):
        gest_df = combined_df[combined_df[label_col] == gest]
        sns.scatterplot(
            data=gest_df,
            x="PC1",
            y="PC2",
            hue=user_col,
            style="Split",
            markers=True,
            alpha=0.8,
            s=80,
            ax=axes2[0, i],
            palette="tab10"
        )
        axes2[0, i].set_title(f"Gesture: {gest}", fontsize=12, fontweight="bold")
        axes2[0, i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        fig2.savefig(os.path.join(output_dir, "pca_train_test_gesture_breakdown.png"), bbox_inches="tight")
    plt.show()
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")


# ==========================================
# 4. Model Evaluation & Metrics
# ==========================================

def plot_confusion_matrix(
    cm: np.ndarray,
    save_fig: bool = True,
    labels: list = None,
    output_dir: str = "figures/evaluation",
    filename: str = "confusion_matrix.png"
):
    """Plots the confusion matrix of the model's performance."""
    plt.figure(figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.show()
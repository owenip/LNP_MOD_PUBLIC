#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
# Force matplotlib to use a non-GUI backend to avoid Qt errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require X server
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys

# Add the project root to the path if needed to import constants
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lnp_mod.config.constants import POST_PROCESS_LABEL_COLORS, POST_PROCESSING_CATEGORIES


def parse_args():
    parser = argparse.ArgumentParser(description='Generate comparison plots for DSC scores across models')
    parser.add_argument('--csv_files', nargs='+', required=True,
                       help='List of DSC CSV files (one per model) to compare')
    parser.add_argument('--model_names', nargs='+', required=True,
                       help='Names of the models corresponding to the CSV files')
    parser.add_argument('--output_dir', default='dsc_comparison_plots',
                       help='Directory to save output plots')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                       help='Figure size for plots (width, height)')
    parser.add_argument('--dpi', type=int, default=500,
                       help='DPI for output plots')
    return parser.parse_args()


def load_dsc_data(csv_files: List[str], model_names: List[str]) -> pd.DataFrame:
    """
    Load DSC data from multiple CSV files and combine into a single DataFrame.
    
    Args:
        csv_files: List of paths to DSC CSV files
        model_names: List of model names corresponding to the CSV files
    
    Returns:
        Combined DataFrame with all DSC data and model information
    """
    if len(csv_files) != len(model_names):
        raise ValueError("Number of CSV files must match number of model names")
    
    all_data = []
    
    for csv_file, model_name in zip(csv_files, model_names):
        df = pd.read_csv(csv_file)
        df['model'] = model_name
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


def generate_box_plot(df: pd.DataFrame, output_path: str, model_palette: dict, figsize: Tuple[int, int] = (12, 8), dpi: int = 300) -> None:
    """
    Generate a box plot comparing DSC scores across different models.
    
    Args:
        df: DataFrame containing DSC scores and model information
        output_path: Path to save the output plot
        model_palette: Dictionary mapping model names to colors
        figsize: Figure size (width, height)
        dpi: DPI for output image
    """
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.boxplot(x='model', y='dsc', data=df, palette=model_palette, fliersize=2, flierprops={'alpha': 0.3})
    
    # Add individual data points with jitter
    # sns.stripplot(x='model', y='dsc', data=df, size=3, color='.3', alpha=0.3, jitter=True)
    
    # Add mean values as text
    for i, model in enumerate(df['model'].unique()):
        mean_dsc = df[df['model'] == model]['dsc'].mean()
        plt.text(i, df['dsc'].max() + 0.02, f'Mean: {mean_dsc:.4f}', 
                 horizontalalignment='center', fontsize=16)
    
    # Customize plot
    plt.title('DSC Score Comparison Across Models', fontsize=20)
    plt.xlabel('Model', fontsize=16)
    plt.ylabel('Dice Similarity Coefficient (DSC)', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def generate_category_box_plot(df: pd.DataFrame, output_path: str, model_palette: dict, figsize: Tuple[int, int] = (16, 10), dpi: int = 300) -> None:
    """
    Generate a box plot comparing DSC scores across different models, grouped by category.
    
    Args:
        df: DataFrame containing DSC scores, category and model information
        output_path: Path to save the output plot
        model_palette: Dictionary mapping model names to colors
        figsize: Figure size (width, height)
        dpi: DPI for output image
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Define specific category order
    category_order = [
        "Bleb with mRNA",
        "Oil Core",
        "Liposomal LNP",
        "Liposome",
        "Other LNP",
        "Not Fully Visible LNP",
        "mRNA",
        "Oil Droplet"
    ]
    
    # Create color mapping for categories
    category_colors = {}
    for category_id, category_name in POST_PROCESSING_CATEGORIES.items():
        if category_name in category_order:
            category_colors[category_name] = POST_PROCESS_LABEL_COLORS[category_id]
    
    # Make a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Convert category column to categorical data type with specific order
    # This determines the order of categories on the x-axis
    plot_df['category'] = pd.Categorical(
        plot_df['category'],
        categories=category_order,
        ordered=True
    )
    
    # Sort dataframe by the ordered category
    plot_df = plot_df.sort_values('category')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot grouped by category with smaller, more transparent outliers
    box_plot = sns.boxplot(
        x='category', 
        y='dsc', 
        hue='model', 
        data=plot_df, 
        ax=ax, 
        palette=model_palette,
        fliersize=2,  # Make outlier points smaller (default is 5)
        flierprops={'alpha': 0.3}  # Make outliers transparent
    )
    
    # Set background colors for each category
    # Get the x-axis positions of the categories
    category_positions = {}
    for i, category in enumerate(category_order):
        if category in plot_df['category'].values:
            category_positions[category] = i
    
    # Get the width of a box
    box_width = 0.8  # This is the default width
    
    # Add colored background for each category
    for category, pos in category_positions.items():
        if category in category_colors:
            # Draw a rectangle with the category color
            rect = plt.Rectangle(
                (pos - box_width/2, 0),  # (x, y)
                box_width,                # width
                1.0,                      # height (covering the entire plot)
                alpha=0.15,               # transparency
                color=category_colors[category],
                zorder=0                  # put it behind other elements
            )
            ax.add_patch(rect)
    
    # Customize plot
    plt.title('DSC Score Comparison by Class and Model', fontsize=20)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Dice Similarity Coefficient (DSC)', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Update legend order
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ['SAM1', 'SAM2.1', 'SAM1 finetuned', 'SAM2.1 finetuned']
    legend_loc = 'lower left'
    box_to_anchor = (0.01, 0.01)
    order = [labels.index(name) for name in desired_order if name in labels]
    if order:
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title='Model', bbox_to_anchor=box_to_anchor, loc=legend_loc, fontsize=14)
    else:
        plt.legend(title='Model', bbox_to_anchor=box_to_anchor, loc=legend_loc, fontsize=14)
    # Rotate category labels for better readability
    plt.xticks(rotation=0, ha='center', fontsize=14)
    
    # Set y-axis limits to 0-1
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and combine DSC data
    df = load_dsc_data(args.csv_files, args.model_names)
    
    # Define a consistent color palette for models
    model_names = args.model_names
    model_palette = sns.color_palette('viridis', n_colors=len(model_names))
    model_palette_dict = dict(zip(model_names, model_palette))
    
    # Generate comparison plots
    generate_box_plot(
        df, 
        str(output_dir / 'overall_dsc_boxplot.svg'), 
        model_palette=model_palette_dict,
        figsize=tuple(args.figsize), 
        dpi=args.dpi
    )

    generate_category_box_plot(
        df, 
        str(output_dir / 'category_dsc_boxplot.svg'), 
        model_palette=model_palette_dict,
        figsize=(args.figsize[0] + 4, args.figsize[1]), 
        dpi=args.dpi
    )

    
    print(f"DSC comparison plots generated in {args.output_dir}/")
    print(f"- Overall DSC box plot: {output_dir / 'overall_dsc_boxplot.svg'}")
    print(f"- Category DSC box plot: {output_dir / 'category_dsc_boxplot.svg'}")


if __name__ == "__main__":
    main() 
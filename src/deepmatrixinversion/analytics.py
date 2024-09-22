#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analytics.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def plot_exp_vs_pred(inverselst_true: list, inverselst_pred: list, save_path: str = None):
    """
    Plot experimental vs predicted inverted matrix values and calculate metrics.

    Args:
    inverselst_true (list): List of true inverted matrices.
    inverselst_pred (list): List of predicted inverted matrices.

    Returns:
    None
    """
    # Input validation
    if len(inverselst_true) != len(inverselst_pred):
        raise ValueError("The lengths of true and predicted lists must be equal.")

    # Flatten and convert to numpy arrays in one step
    ytrue = np.array([num for matrix in inverselst_true for row in matrix for num in row])
    ypred = np.array([num for matrix in inverselst_pred for row in matrix for num in row])


    # Calculate metrics
    r2 = r2_score(ytrue, ypred)
    mse = mean_squared_error(ytrue, ypred)
    mae = mean_absolute_error(ytrue, ypred)

    # Perform linear regression
    reg = LinearRegression().fit(ytrue.reshape(-1, 1), ypred)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Print metrics and regression equation
    print(f"R2: {r2:.4f} MSE: {mse:.4f} MAE: {mae:.4f}")
    print(f"Regression equation: y = {slope:.4f}x + {intercept:.4f}")

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(ytrue, ypred, s=3, alpha=0.5, label='Data points')
    
    # Plot regression line
    x_range = np.linspace(ytrue.min(), ytrue.max(), 100)
    y_pred = slope * x_range + intercept
    plt.plot(x_range, y_pred, color='red', label=f'Regression line (y = {slope:.2f}x + {intercept:.2f})')

    # Add diagonal line for perfect predictions
    min_val, max_val = min(ytrue.min(), ypred.min()), max(ytrue.max(), ypred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', lw=2, label='Perfect prediction')

    plt.xlabel("Experimental inverted matrix values")
    plt.ylabel("Predicted inverted matrix values")
    plt.title("Experimental vs Predicted Inverted Matrix Values")
    plt.legend()

    # Add text box with metrics
    textstr = f'R² = {r2:.4f}\nMSE = {mse:.4f}\nMAE = {mae:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_diff(inverselst_true: list, inverselst_pred: list, save_path: str = None):
    fig, axes = plt.subplots(1, len(inverselst_true), figsize=(5*len(inverselst_true), 5))
    for i, (true, pred) in enumerate(zip(inverselst_true, inverselst_pred)):
        diff = true - pred
        im = axes[i].imshow(diff, cmap='RdYlGn')
        axes[i].set_title(f'Matrix {i+1}')
        for (j,k), value in np.ndenumerate(diff):
            axes[i].text(k, j, f'{value:.2f}', ha='center', va='center')
        fig.colorbar(im, ax=axes[i])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


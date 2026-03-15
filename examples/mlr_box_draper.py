#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLR Application Example: Box-Draper Chemical System

This script demonstrates using the Neural Network matrix inversion for 
Ordinary Least Squares (OLS) regression on the Box-Draper dataset.
It compares coefficients obtained via NumPy vs. DeepMatrixInversion.

Model: y = b0 + b1x1 + b2x2 + b3x3 + b12x1x2 + b13x1x3 + b23x2x3 + b11x1^2 + b22x2^2 + b33x3^2
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from deepmatrixinversion.inference import MatrixInversionInference

def plot_beta_comparison(beta_np, beta_nn, terms, save_path=None):
    """
    Plot comparison of beta coefficients between NumPy and NN.
    """
    x = np.arange(len(terms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, beta_np, width, label='NumPy (Exact)', color='skyblue')
    rects2 = ax.bar(x + width/2, beta_nn, width, label='Neural Network', color='salmon')

    ax.set_ylabel('Coefficient Value')
    ax.set_title('Beta Coefficients Comparison: NumPy vs. Neural Network Inversion')
    ax.set_xticks(x)
    ax.set_xticklabels(terms)
    ax.legend()

    # Add error values as text
    for i in range(len(terms)):
        diff = abs(beta_np[i] - beta_nn[i])
        ax.text(x[i], max(beta_np[i], beta_nn[i]) + 0.5, f"err:{diff:.2e}", 
                ha='center', va='bottom', fontsize=8, rotation=45)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def get_box_draper_data():
    """
    Data from Box & Draper (11.3): A Consecutive Chemical System
    Columns: Time (x1), Catalyst (x2), Temperature (x3), Yield (Y)
    """
    data = np.array([
        [5.0, 25.0, 162.0, 45.9],
        [8.0, 25.0, 162.0, 53.3],
        [5.0, 30.0, 162.0, 57.5],
        [8.0, 30.0, 162.0, 58.8],
        [5.0, 25.0, 172.0, 60.6],
        [8.0, 25.0, 172.0, 58.0],
        [5.0, 30.0, 172.0, 58.6],
        [8.0, 30.0, 172.0, 52.4],
        [6.5, 27.5, 167.0, 56.9],
        [6.5, 27.5, 177.0, 55.4],
        [6.5, 27.5, 157.0, 46.9],
        [6.5, 32.5, 167.0, 57.5],
        [6.5, 22.5, 167.0, 55.0],
        [9.5, 27.5, 167.0, 58.9],
        [3.5, 27.5, 167.0, 50.3],
        [6.5, 20.0, 177.0, 61.1],
        [6.5, 20.0, 177.0, 62.9],
        [7.5, 34.0, 160.0, 60.0],
        [7.5, 34.0, 160.0, 60.6]
    ])
    return data[:, :3], data[:, 3]

def expand_features(X):
    """
    Expand [x1, x2, x3] to [1, x1, x2, x3, x1x2, x1x3, x2x3, x1^2, x2^2, x3^2]
    """
    n_samples = X.shape[0]
    X_exp = np.ones((n_samples, 10))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    
    X_exp[:, 1] = x1
    X_exp[:, 2] = x2
    X_exp[:, 3] = x3
    X_exp[:, 4] = x1 * x2
    X_exp[:, 5] = x1 * x3
    X_exp[:, 6] = x2 * x3
    X_exp[:, 7] = x1**2
    X_exp[:, 8] = x2**2
    X_exp[:, 9] = x3**2
    
    return X_exp

def pad_matrix(M, msize):
    """
    Pad matrix M to be a multiple of msize using identity padding.
    """
    size = M.shape[0]
    new_size = ((size + msize - 1) // msize) * msize
    if new_size == size:
        return M
    
    M_padded = np.eye(new_size)
    M_padded[:size, :size] = M
    return M_padded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to 3x3 trained model directory")
    parser.add_argument("--mode", type=str, default="numpy", choices=["nn", "numpy"], 
                        help="Inversion mode")
    parser.add_argument("--plotout", type=str, default=None, help="Save plot to file (e.g., plot.png)")
    args = parser.parse_args()

    # 1. Load and Scale Data
    X_raw, Y = get_box_draper_data()
    
    # Scale to [-1, 1] for numerical stability and NN compatibility (Coded variables)
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_coded = 2 * (X_raw - X_min) / (X_max - X_min) - 1
    
    # 2. Expand Features
    X = expand_features(X_coded)
    
    # 3. Construct Normal Equations
    XTX = X.T @ X
    XTY = X.T @ Y
    
    print(f"Constructed XTX matrix of size {XTX.shape}")
    
    # 4. Perform Inversion
    if args.mode == "numpy":
        print("Using NumPy linalg.inv...")
        XTX_inv = np.linalg.inv(XTX)
    else:
        print(f"Using DeepMatrixInversion (NN) from {args.model}...")
        inference = MatrixInversionInference(models_path=args.model, invert_mode="nn")
        
        # Pad 10x10 to 12x12 for Schur complement with 3x3 base blocks
        XTX_padded = pad_matrix(XTX, inference.nn.msize)
        print(f"Padded XTX to {XTX_padded.shape} for recursion...")
        
        XTX_inv_padded = inference.invert([XTX_padded])[0]
        XTX_inv = XTX_inv_padded[:10, :10]
        
    # 5. Calculate Beta Coefficients
    beta = XTX_inv @ XTY
    
    # 6. Compare with baseline if we ran NN or if plot is requested
    terms = ["b0", "b1", "b2", "b3", "b12", "b13", "b23", "b11", "b22", "b33"]
    if args.mode == "nn":
        beta_np = np.linalg.inv(XTX) @ XTY
        diff = np.abs(beta - beta_np)
        
        print("\nComparison of Beta Coefficients:")
        print(f"{'Term':<10} | {'NumPy':>12} | {'NN':>12} | {'Diff':>12}")
        print("-" * 55)
        for i, term in enumerate(terms):
            print(f"{term:<10} | {beta_np[i]:12.4f} | {beta[i]:12.4f} | {diff[i]:12.4e}")
            
        print(f"\nMean Absolute Difference: {np.mean(diff):.4e}")
        
        if args.plotout:
            plot_beta_comparison(beta_np, beta, terms, args.plotout)
    else:
        print("\nBeta Coefficients (NumPy):")
        for i, term in enumerate(terms):
            print(f"{term:<10}: {beta[i]:12.4f}")
        
        if args.plotout:
            print("Note: Plot comparison only available in --mode nn")

if __name__ == "__main__":
    main()

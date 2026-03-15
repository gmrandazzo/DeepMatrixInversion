#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple OLS Example: 3x3 Direct Matrix Inversion

This script demonstrates using the Neural Network matrix inversion for 
a simple 3-parameter OLS regression. This fits exactly into the 
base 3x3 model without needing Schur complement recursion or padding.

Model: y = b0 + b1x1 + b2x2
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from deepmatrixinversion.inference import MatrixInversionInference

def generate_simple_data(n_samples=50):
    """
    Generate synthetic data for y = 5 + 2x1 - 3x2 + noise
    """
    np.random.seed(42)
    x1 = np.random.uniform(-1, 1, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)
    
    y = 5 + 2*x1 - 3*x2 + noise
    
    # Construct X matrix [1, x1, x2]
    X = np.ones((n_samples, 3))
    X[:, 1] = x1
    X[:, 2] = x2
    
    return X, y

def plot_simple_beta(beta_np, beta_nn, terms, save_path=None):
    x = np.arange(len(terms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, beta_np, width, label='NumPy (Exact)', color='lightgray')
    ax.bar(x + width/2, beta_nn, width, label='Neural Network', color='green', alpha=0.7)
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Simple OLS (3x3): NumPy vs. Neural Network')
    ax.set_xticks(x)
    ax.set_xticklabels(terms)
    ax.legend()
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to 3x3 trained model directory")
    parser.add_argument("--mode", type=str, default="numpy", choices=["nn", "numpy"], 
                        help="Inversion mode")
    parser.add_argument("--plotout", type=str, default=None, help="Save plot to file")
    args = parser.parse_args()

    # 1. Generate Data
    X, Y = generate_simple_data()
    
    # 2. Construct Normal Equations
    XTX = X.T @ X
    XTY = X.T @ Y
    
    print(f"Constructed XTX matrix of size {XTX.shape}")
    
    # 3. Perform Inversion
    if args.mode == "numpy":
        print("Using NumPy linalg.inv...")
        XTX_inv = np.linalg.inv(XTX)
    else:
        print(f"Using DeepMatrixInversion (NN) from {args.model}...")
        inference = MatrixInversionInference(models_path=args.model, invert_mode="nn")
        XTX_inv = inference.invert([XTX])[0]
        
    # 4. Calculate Beta Coefficients
    beta = XTX_inv @ XTY
    terms = ["Intercept", "b1 (x1)", "b2 (x2)"]
    
    # 5. Comparison
    if args.mode == "nn":
        beta_np = np.linalg.inv(XTX) @ XTY
        diff = np.abs(beta - beta_np)
        
        print("\nCoefficient Comparison:")
        for i, term in enumerate(terms):
            print(f"{term:<12}: NumPy={beta_np[i]:.4f}, NN={beta[i]:.4f}, Diff={diff[i]:.2e}")
            
        if args.plotout:
            plot_simple_beta(beta_np, beta, terms, args.plotout)
    else:
        print("\nBeta Coefficients (NumPy):")
        for i, term in enumerate(terms):
            print(f"{term:<12}: {beta[i]:.4f}")

if __name__ == "__main__":
    main()

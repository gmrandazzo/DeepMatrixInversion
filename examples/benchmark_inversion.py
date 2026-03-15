#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: NumPy vs. Neural Network Recursive Inversion

This script measures the execution time for inverting matrices of increasing 
sizes (multiples of the base model size) using both methods.
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from deepmatrixinversion.inference import MatrixInversionInference

def benchmark(model_path, min_k=1, max_k=10, repeats=3):
    """
    Benchmark inversion for matrices of size (k * msize) x (k * msize)
    """
    # Setup inference
    # Note: We use invert_mode="nn" for NN and a separate timer for numpy
    inference_nn = MatrixInversionInference(models_path=model_path, invert_mode="nn")
    msize = inference_nn.nn.msize
    
    results = {
        "sizes": [],
        "nn_times": [],
        "np_times": []
    }
    
    for k in range(min_k, max_k + 1):
        size = k * msize
        results["sizes"].append(size)
        print(f"Benchmarking size {size}x{size}...", end=" ", flush=True)
        
        # Generate random matrix
        mx = np.random.uniform(-1, 1, (size, size))
        # Ensure it's reasonably invertible
        mx = mx + np.eye(size) * size 
        
        # 1. Benchmark NumPy
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = np.linalg.inv(mx)
        t_np = (time.perf_counter() - t0) / repeats
        results["np_times"].append(t_np)
        
        # 2. Benchmark Neural Network (Recursive)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = inference_nn.invert([mx])
        t_nn = (time.perf_counter() - t0) / repeats
        results["nn_times"].append(t_nn)
        
        print(f"NP: {t_np:.5f}s, NN: {t_nn:.5f}s")
        
    return results

def plot_performance(results, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(results["sizes"], results["nn_times"], 'o-', label='Neural Network (Recursive)', color='salmon')
    plt.plot(results["sizes"], results["np_times"], 's-', label='NumPy (MKL/OpenBLAS)', color='skyblue')
    
    plt.yscale('log')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.title('Execution Time: NumPy vs. Neural Network Inversion')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Performance plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--max_k", type=int, default=10, help="Max multiplier of base size (default 10)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions per size")
    parser.add_argument("--plotout", type=str, default="benchmark_results.png", help="Save plot to file")
    args = parser.parse_args()

    results = benchmark(args.model, max_k=args.max_k, repeats=args.repeats)
    plot_performance(results, args.plotout)

if __name__ == "__main__":
    main()

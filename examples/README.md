# DeepMatrixInversion Application Examples

This directory contains practical applications of the Neural Network matrix inversion method.

## 1. Simple OLS Regression (3x3 Direct)

`simple_ols_3x3.py` demonstrates a straightforward 3-parameter Ordinary Least Squares (OLS) problem. This example uses a synthetic dataset to fit a linear model:
$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon$

Since the resulting Normal Equation matrix ($X^T X$) is exactly **3x3**, it is inverted directly by the neural network without requiring Schur complement recursion or padding.

![Simple OLS Comparison](simple_ols_plot.png)
*(Note: Run the script with --plotout to generate this image locally)*

### How to Run:
```bash
# 1. Train a 3x3 model (if not already done)
dmxtrain --msize 3 --epochs 5000 --mout ../Model_3x3

# 2. Run the simple example
python3 simple_ols_3x3.py --mode nn --model ../Model_3x3_YYYYMMDDHHMMSS --plotout simple_ols.png
```

---

## 2. Advanced MLR: Box-Draper Chemical System

We use the dataset from the following textbook:

> **Box, G. E. P. and Draper, N. R. (2007). Response Surfaces, Mixtures, and Ridge Analyses, Second Edition. Wiley.** 
> [DOI: 10.1002/0470072768](https://onlinelibrary.wiley.com/doi/book/10.1002/0470072768)
> **Example 11.3, pages 348–353.** 

The system studies the yield of a consecutive chemical reaction based on three variables:
-   $x_1$: Time (hours)
-   $x_2$: Catalyst (%)
-   $x_3$: Temperature (°C)

The goal is to fit a second-order polynomial model:
$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_3 + \beta_{12}x_1x_2 + \beta_{13}x_1x_3 + \beta_{23}x_2x_3 + \beta_{11}x_1^2 + \beta_{22}x_2^2 + \beta_{33}x_3^2$

## Implementation Details

### 1. Coded Variables
In response surface methodology, it is standard practice to use "coded" variables. We scale the raw inputs to the range $[-1, 1]$. This is mathematically beneficial for the OLS estimation and aligns the data range with the expectations of our Neural Network model.

### 2. Normal Equations
OLS solves for $\beta$ using the formula:
$\beta = (X^T X)^{-1} X^T Y$

The matrix $X^T X$ for this problem is **10x10**. 

### 3. Recursive Inversion with Padding
Since our base neural network model is trained on **3x3** matrices, we cannot invert the 10x10 matrix directly. The application handles this by:
1.  **Padding**: The 10x10 matrix is padded with an identity block to become **12x12**.
2.  **Recursive Schur Complement**: The 12x12 matrix is recursively inverted using the Schur complement method until it reaches the 3x3 base blocks handled by the neural network.

## How to Run

1.  **Train the Model**:
    Ensure you have a trained 3x3 model ensemble.
    ```bash
    dmxtrain --msize 3 --epochs 5000 --mout ./Model_3x3
    ```

2.  **Run the Example**:
    Provide the path to your trained model directory.
    ```bash
    python3 mlr_box_draper.py --mode nn --model ../Model_3x3_YYYYMMDDHHMMSS --plotout comparison_plot.png
    ```

## Results Visualization

The script generates a comparison between the "exact" coefficients (NumPy) and the neural network's approximation.

![Beta Coefficients Comparison](comparison_plot.png)
*(Note: Run the script with --plotout to generate this image locally)*

The comparison shows how well the neural network, even when trained on tiny 3x3 matrices, can be used to solve much larger, real-world regression problems through recursive decomposition.

---

## 3. Benchmark: Execution Time Comparison

`benchmark_inversion.py` compares the execution speed of the **Neural Network (Recursive)** method vs. the highly optimized **NumPy (LAPACK/BLAS)** implementation. 

The script benchmarks matrices of size $N \times N$, where $N$ increases as multiples of the model's base size (e.g., 3, 6, 9, 12, ...). 

### How to Run:
```bash
python3 benchmark_inversion.py --model ../Model_3x3_YYYYMMDDHHMMSS --max_k 10 --plotout benchmark_results.png
```

![Benchmark Performance](benchmark_results.png)
*(Note: Run the script to generate this plot. Given that the NN approach is recursive and implemented in high-level Python, NumPy will typically be faster for standard sizes. This benchmark demonstrates the computational scaling of the recursive approach.)*

# Official Models Directory

This directory stores pretrained neural network ensembles for matrix inversion. 

## Model Architecture

The core of DeepMatrixInversion is a deep **Residual Network (ResNet)** specifically optimized for the numerical precision required by linear algebra operations.

### Key Features:
-   **Residual Blocks:** The model uses 7 residual layers. Each block learns the *difference* between the current state and the target inverse, making the training more stable and allowing for deeper architectures.
-   **DenseNet-style Growth:** Within each residual block, we implement 4 sub-layers that use concatenation (rather than just addition) to preserve high-frequency features from previous layers before projecting back to the skip-connection dimension.
-   **Zero-Initialization:** The final projection layer of each residual block is initialized with zeros. This ensures that at the start of training, each residual block acts as an identity mapping, allowing gradients to flow unimpeded through the skip connections.
-   **Ensemble Averaging:** By default, the tool trains and uses an ensemble of models (e.g., 3-5 independent networks). During inference, their predictions are averaged to reduce individual variance and improve numerical stability.

## Mathematical Foundation

Unlike traditional regression models that might use Mean Squared Error (MSE), our model is trained using a specialized **Frobenius Loss (`floss`)**.

### The Loss Function
The network is optimized to satisfy the fundamental definition of a matrix inverse:
$$ \text{loss} = || I - A \cdot A^{-1} ||_F $$

Where:
-   $A$ is the input matrix.
-   $A^{-1}$ is the predicted inverse.
-   $I$ is the identity matrix.
-   $||\cdot||_F$ is the Frobenius Norm.

By directly optimizing for the identity property, the model learns the structural relationships of matrix elements rather than just fitting individual values.

## File Structure

Each model ensemble is stored in a subdirectory:

### Recommended Structure:
```
models/
└── resnet_7layers_3x3_20240920182800_v1/
    ├── config.toml
    ├── 1.keras
    ├── 2.keras
    └── 3.keras
```

-   **`*.keras` files:** The saved weights and architecture for each member of the ensemble.
-   **`config.toml`:** Metadata containing the matrix size (`msize`), scaling factors, and normalization ranges used during training.

## Usage
To load a model from this directory for inference:
```bash
dmxinvert --model models/resnet_7layers_3x3_20240920182800_v1 --inputmx data.csv --inverseout results.csv
```

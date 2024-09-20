# DeepMatrixInversion

Invert matrix using a neural network.

## Challenges of Matrix Inversion with Neural Networks
Inverting matrices presents unique challenges for neural networks, primarily due to inherent limitations in performing precise arithmetic operations such as multiplication and division on activations. Traditional dense networks often need help with these tasks, as they are not explicitly designed to handle the complexities involved in matrix inversion. Experiments conducted with simple dense neural networks have shown significant difficulties achieving accurate matrix inversions. Despite various attempts to optimize the architecture and training process, the results often need improvement. However, transitioning to a more complex architecture—a 7-layer Residual Network (ResNet)—can lead to marked improvements in performance.

## The ResNet Advantage
The ResNet architecture, known for its ability to learn deep representations through residual connections, has proven effective in tackling matrix inversion. With millions of parameters, this network can capture intricate patterns within the data that simpler models cannot. However, this complexity comes at a cost: substantial training data are required for effective generalization.

![Predicted Inverted Matrix](https://github.com/gmrandazzo/DeepMatrixInversion/blob/master/Results/Figure_1_Inverted_Matrix_Predicted_3x3.png)
*Figure 1: Visualization of a neural network predicted inverted matrix for a set of matrices 3x3 never seen in the dataset*


### Loss Function

To evaluate the performance of the neural network in predicting matrix inversions, a specific loss function is employed:

$$
\text{loss} = || I - AA^{-1} ||
$$

In this equation:

- $A$ represents the original matrix.
- $A^{-1}$ denotes the predicted inverse of matrix $A$.
- $I$ is the identity matrix.
- || || is the Frobenius Norm

The goal is to minimize the difference between the identity matrix and the product of the original matrix and its predicted inverse.
This loss function effectively measures how close the predicted inverse is to being accurate.

Additionally, if $y_{\text{true}}$ is defined as the true inverse and $y_{\text{pred}}$ as the predicted inverse, this loss function can also be interpreted as:

$$
\text{loss} = || y_{\text{true}} - y_{\text{pred}} ||
$$

This loss function offers distinct advantages over traditional loss functions such as Mean Squared Error (MSE) or Mean Absolute Error (MAE).

1. Direct Measurement of Inversion Accuracy
The primary goal of matrix inversion is to ensure that the product of a matrix and its inverse yields the identity matrix. The loss function directly captures this requirement by measuring the deviation from the identity matrix. In contrast, MSE and MAE focus on the differences between predicted values and true values without explicitly addressing the fundamental property of matrix inversion.

2. Emphasis on Structural Integrity
By using a loss function that evaluates how close the product AA−1AA−1 is to II, it emphasizes maintaining the structural integrity of the matrices involved. This is particularly important in applications where preserving linear relationships is crucial. Traditional loss functions like MSE and MAE do not account for this structural aspect, potentially leading to solutions that minimize error but fail to satisfy the mathematical requirements of matrix inversion.

3. Applicability to Non-Singular Matrices
This loss function inherently assumes that the matrices being inverted are non-singular (i.e., invertible). In scenarios where singular matrices are present, traditional loss functions might yield misleading results since they do not account for the impossibility of obtaining a valid inverse. The proposed loss function highlights this limitation by producing larger errors when attempting to invert singular matrices.

### The Problem of Singular Matrices

One significant limitation when using neural networks for matrix inversion is their inability to handle singular matrices effectively. A singular matrix does not have an inverse; thus, any attempt by a neural network to predict an inverse for such matrices will yield incorrect results. In practice, if a singular matrix is presented during training or inference, the network may still output a result, but this output will not be valid or meaningful. This limitation underscores the importance of ensuring that training data consists of non-singular matrices whenever possible.

![Singular Matrix Prediction](https://github.com/gmrandazzo/DeepMatrixInversion/blob/master/Results/Figure_2_Singular_Matrix_Predicted_3x3.png)
*Figure 2: Comparison of model prediction for singular matrices versus pseudoinversions. Note that the model will produce results regardless of matrix singularity.*

## Data Requirements and Overfitting
Research indicates that a ResNet model can memorize a good amount of samples without significant loss of accuracy. However, increasing the dataset size to 10 million samples may lead to severe overfitting. This overfitting occurs despite the large volume of data, highlighting that simply increasing dataset size does not guarantee improved generalization for complex models. To address this challenge, a continuous data generation strategy can be adopted. Instead of relying on a static dataset, samples can be generated on the fly and fed to the network as they are created. This approach, which is crucial in mitigating overfitting, not only provides a diverse range of training examples but also ensures that the model is exposed to a constantly evolving dataset.

## Conclusion
In summary, while matrix inversion is inherently challenging for neural networks due to limitations in arithmetic operations, leveraging advanced architectures like ResNet can yield better results. However, careful consideration must be given to data requirements and overfitting risks. Continuously generating training samples can enhance the model's learning process and improve performance in matrix inversion tasks. This version maintains an impersonal tone while discussing the challenges and strategies in training neural networks for matrix inversion.

License
=======

DeepMatrixInversion is distributed under LGPLv3 license

To know more in details how the licens work please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/lgpl-3.0.html"

DeepMatrixInversion is currently property of Giuseppe Marco Randazzo.


Dependencies
============

- python version  >= 3.9
- numpy
- matplotlib
- scikit-learn
- tensorflow
- toml


Installation
============

To install the DeepMatrixInversion repository, you can choose between using poetry or pipx
Below are the instructions for both methods.

#### Poetry
1. Clone the Repository: Use the following command to clone the repository from GitHub.

```
git clone https://github.com/gmrandazzo/DeepMatrixInversion.git
```

2. Navigate to the Directory: Change into the directory of the cloned repository.

```
cd DeepMatrixInversion
```

3. Install Dependencies: Use Poetry to install the required dependencies for the project.

```
python3 -m venv .venv
. .venv/bin/activate
pip install poetry
poetry install
```

This will set up your environment with all necessary packages to run DeepMatrixInversion.

#### Using pipx

If you prefer to use pipx, which allows you to install Python applications in isolated environments, follow these steps:

1. Ensure pipx is Installed: First, make sure you have pipx installed on your system. If you haven't installed it yet, you can do so using one of the following commands:

- Using pip:
```
python3 -m pip install --user pipx
```

- Using apt (for Debian-based systems):

```
apt-get install pipx
```

- Using Homebrew (for macOS):

```
brew install pipx
```

- Using dnf (for Fedora-based systems):
```
sudo  dnf install pipx
```

2. Install DeepMatrixInversion from GitHub: Use the following command to install the package directly from the GitHub repository:

pipx install git+https://github.com/gmrandazzo/DeepMatrixInversion.git


Usage
=====

To train a model that can perform matrix inversion, you will use the dmxtrain command.
This command allows you to specify various parameters that control the training process,
such as the size of the matrices, the range of values, and the training duration.


```bash
dmxtrain --msize <matrix_size> --rmin <min_value> --rmax <max_value> --epochs <number_of_epochs> --batch_size <size_of_batches> --n_repeats <number_of_repeats> --mout <output_model_path>
```

### Example:

```
 dmxtrain --msize --rmin -1 --rmax 1 --epochs 5000 --batch_size 1024 --n_repeats 3 --mout ./Model_3x3
```

#### Prameters
```
    --msize <matrix_size>: Specifies the size of the square matrices to be generated for training. For example, 3 for 3x3 matrices.
    --rmin <min_value>: Sets the minimum value for the random elements in the matrices. For instance, -1 will allow negative values.
    --rmax <max_value>: Sets the maximum value for the random elements in the matrices. For example, 1 will limit values to a maximum of 1.
    --epochs <number_of_epochs>: Defines how many epochs (complete passes through the training dataset) to run during training. A higher number typically leads to better performance; in this case, 5000.
    --batch_size <size_of_batches>: Determines how many samples are processed before the model is updated. A batch size of 1024 means that 1024 samples are used in each iteration.
    --n_repeats <number_of_repeats>: Indicates how many times to repeat the training process with different random seeds or initializations. This can help ensure robustness; for instance, repeating 3 times.
    --mout <output_model_path>: Specifies where to save the trained model. In this example, it saves to ./Model_3x3.
```

Once you have trained your model, you can use it to perform matrix inversion on new input matrices.
The command for inference is dmxinvert, which takes an input matrix and outputs its inverse.

```
dmxinvert --inputmx <input_matrix_file> --inverseout <output_csv_file> --model <model_path>
```

### Example:

```
dmxinvert --inputmx input_matrix.csv --inverseout output_inverse.csv --model ./Model_3x3_*
```

#### Parameters
```
    --inputmx <input_matrix_file>: Specifies the path to the input matrix file that you want to invert. This file should contain a valid matrix format (e.g., CSV).
    --inverseout <output_csv_file>: Indicates where to save the resulting inverted matrix. The output will be saved in CSV format.
    --model <model_path>: Provides the path to the trained model that will be used for performing the inversion.
```


#### Example Input Matrix Format

The input matrix file should be formatted as follows:

```
0.24077047370124594,-0.5012474139608847,-0.5409542929032876
-0.6257864520097793,-0.030705148203584942,-0.13723920334288975
-0.48095686716222064,0.19220406568380666,-0.34750000491973854
END
0.4575368007107925,0.9627977617090073,-0.4115240560547333
0.5191433428806012,0.9391491187187144,-0.000952683255491138
-0.17757763984424968,-0.7696584771443977,-0.9619759413623306
END
-0.49823271153034154,0.31993947803488587,0.9380291202366384
0.443652116558352,0.16745965310481048,-0.267270356721347
0.7075720067281346,-0.3310912886946993,-0.12013367141105102
END
```

Each block of numbers represents a separate matrix followed by an END
marker indicating the end of that matrix.

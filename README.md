# DeepMatrixInversion

Invert matrix using a neural network.


License
============

DeepMatrixInversion is distributed under LGPLv3 license

To know more in details how the licens work please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/lgpl-3.0.html"

DeepMatrixInversion is currently property of Giuseppe Marco Randazzo.


Dependencies
============

- python version  3.9
- numpy
- sklearn
- matplotlib
- tensorflow


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

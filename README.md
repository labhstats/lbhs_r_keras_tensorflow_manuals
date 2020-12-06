# labhstat's: R Keras Tensorflow (GPU) code and scripts
Repository where I back up my learning process of creating or implementing neural networks via keras in R.

To be continued as I have time.

## Data sets:
- boligpriser_tromso_desember_2020.csv: A small (N = 140) set of data with house pricings. Mix of continous, integers and nominal categories.

## Scripts:
- bolig_TOS_2020.R: Tried run of "boligpriser_tromso_desember_2020.csv" data.
- mnist_example.R: Implementation of the standard MNIST example.
- nn_building_blocks.R: Attempt to create blocks and then "generalise" known neural network. Syntax inspired by https://github.com/r-tensorflow/unet/blob/master/R/model.R#L53 to avoid the pipe operator, which becomes confusing(?) when chaining blocks of layers.
- summary_building_block_models.R: Script to check that composition of blocks and layers in "nn_building_blocks.R" actually return a comprehensible structure ready for compiling and fitting.

## Installation of Keras with Tensorflow GPU [Bottom to top instructions]

Keras:
https://cran.r-project.org/web/packages/keras/vignettes/index.html
https://keras.rstudio.com/reference/install_keras.html (Better)

Tensorflow: (Backend to Keras)
https://tensorflow.rstudio.com/installation/
https://www.tensorflow.org/install/gpu#hardware_requirements
Software requirements (Tensorflow)
The following NVIDIA® software must be installed on your system:
NVIDIA® GPU drivers —CUDA® 10.1 requires 418.x or higher.
CUDA® Toolkit —TensorFlow supports CUDA® 10.1 (TensorFlow >= 2.1.0)
CUPTI ships with the CUDA® Toolkit.
cuDNN SDK 7.6 (see cuDNN versions).
(Optional) TensorRT 6.0 to improve latency and throughput for inference on some models.
Windows installs:

Anaconda: (Python 3 for Tensorflow; before running 'install_keras(tensorflow = "gpu")'.)
https://www.anaconda.com/products/individual

cuDNN:
- https://developer.nvidia.com/cudnn
- https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows 
- https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781 [Better guide; remember to put library files from cuDNN zip into CUDA]

CUDA Toolkit all versions: (Installer)
- https://developer.nvidia.com/cuda-toolkit-archive
- https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-microsoft-windows/index.html [10.1 specifically]
- https://docs.nvidia.com/deploy/cuda-compatibility/index.html 

CUDA installation guide:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

Visual Studio: (Requirement for CUDA installer) [Only base version required]
https://visualstudio.microsoft.com/

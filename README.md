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

## Installation of Keras with Tensorflow GPU


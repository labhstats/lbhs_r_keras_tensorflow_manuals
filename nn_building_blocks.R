####
# Building blocks for Neural Networks: Keras & Tensorflow

library(keras)

#####
# Middle blocks
# Attempt to generalize "typical" repeated structures of interest.
#####

##
# Layer block: Dense & Dropout
layer_block_dense_drop = function(object,
                                  units_b = 64,
                                  activation_b = "relu",
                                  rate_b = 0.1){
  require(keras)
  
  x <- object
  
  x <- layer_dense(object = x, units = units_b,activation = activation_b)
  
  x <- layer_dropout(object = x, rate = rate_b)
  
  return(x)
}


##
# Layer block: 2x Conv2D, Maxpool, Dropout
layer_block_2conv2D_maxpool_drop = function(object,
                                            filters_b = 32,
                                            kernel_size_b = c(3,3),
                                            activation_b = "relu",
                                            pool_size_b = c(2,2),
                                            rate_b = 0.25){
  require(keras)
  
  x <- object
  
  x <- layer_conv_2d(object = x, filters = filters_b, kernel_size = kernel_size_b,
                     activation = activation_b)
  
  x <- layer_conv_2d(object = x, filters = filters_b, kernel_size = kernel_size_b,
                     activation = activation_b)
  
  x <- layer_max_pooling_2d(object = x, pool_size = pool_size_b)
  
  x <- layer_dropout(object = x, rate = rate_b)
  
  return(x)
}

#####
# Initial
#####

##
# Layer block: Initialize with Dense & Dropout
layer_init = function(input_shape){
  require(keras)
  #Quite redundant, really. 
  #But it was not this simply originally, but turns out it had to be this simple.
  
  x <- layer_input(shape = input_shape)
  
  return(x)
}

#####
# Ends
#####

##
# Layer end: Force scalar output after dense layer
layer_end_dense_scal_cont = function(object,
                               units_b = 32,
                               activation_b = "relu"){
  require(keras)
  
  x <- object
  
  x <- layer_dense(object = x,units = units_b,activation = activation_b)
  
  x <- layer_dense(object = x, units = 1)
  
  return(x)
}

##
# Layer end: Force scalar output
layer_end_scal_cont = function(object){
  require(keras)
  
  x <- object
  
  x <- layer_dense(object = x, units = 1)
  
  return(x)
}

##
# Layer end: Softmax (Multiple [num_classes > 2] categorical classification)
layer_end_multinom = function(object, num_classes = 10){
  # Layer end: Softmax (Multiple [num_classes > 2] categorical classification)
  
  require(keras)
  
  x <- object
  
  x <- layer_dense(object = x,units = num_classes,activation = "softmax")
  
  return(x)
}

##
# Layer end: Sigmoid (Binary categorical classification)

layer_end_logistic = function(object){
  # Layer end: Sigmoid (Binary categorical classification)
  
  require(keras)
  
  x <- object
  
  x <- layer_dense(object = x,units = 1,activation = "sigmoid")
  
  return(x)
}


#####
# Compositions:
#####

##
# Composition: Feed Forward Neural Network for data.frame regression
ff_nn_df_regression = function(input_layer = layer_input_from_dataset(train_df %>% select(-tot_pris)),
                               spec_reg = spec,
                               width_blocks = c(32,64,32), #Vector of integers
                               block_dropout = 0.1, #Single
                               block_activation = "relu", #Single
                               end_width = 16,
                               end_activation = "relu"){
  # Composition: Feed Forward Neural Network for data.frame regression
  # Inspired by: https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_regression/
  #              Information about how to handle data.frame and get a "spec" also present there.
  
  require(keras)
  
  input = input_layer
  
  x <- input
  
  x <- layer_dense_features(object = x,feature_columns = dense_features(spec_reg))
  
  dense_drop_blocks = length(width_blocks)
  for(i in 1:dense_drop_blocks){
    x <- layer_block_dense_drop(object = x,
                                units_b = width_blocks[i],
                                activation_b = block_activation,
                                rate_b = block_dropout)
  }
  
  output <- layer_end_dense_scal_cont(object = x,units_b = end_width,activation_b = end_activation)
  
  model <- keras_model(inputs = input, outputs = output)
  
  return(model)
}

##
# Composition: MLP for multiple categories
mlp_mult_cat = function(input_shape,
                        num_classes,
                        width_blocks = c(64,64),
                        block_dropout = 0.5,
                        block_activation = "relu"){
  # Composition: MLP for multiple categories
  # Inspired by: https://keras.rstudio.com/articles/sequential_model.html
  
  require(keras)
  
  input <- layer_init(input_shape = input_shape)
  
  x <- input
  
  dense_drop_blocks = length(width_blocks)
  for(i in 1:dense_drop_blocks){
    x <- layer_block_dense_drop(object = x,
                                units_b = width_blocks[i],
                                activation_b = block_activation,
                                rate_b = block_dropout)
  }
  
  
  output <- layer_end_multinom(object = x,num_classes = num_classes)
  
  model <- keras_model(inputs = input,outputs = output)
  
  return(model)
}

##
# Composition: MLP for binary categories
mlp_bin_cat = function(input_shape,
                        width_blocks = c(64,64),
                        block_dropout = 0.5,
                        block_activation = "relu"){
  # Composition: MLP for binary categories
  # Inspired by: https://keras.rstudio.com/articles/sequential_model.html
  
  require(keras)
  
  input <- layer_init(input_shape = input_shape)
  
  x <- input
  
  dense_drop_blocks = length(width_blocks)
  for(i in 1:dense_drop_blocks){
    x <- layer_block_dense_drop(object = x,
                                units_b = width_blocks[i],
                                activation_b = block_activation,
                                rate_b = block_dropout)
  }
  
  
  output <- layer_end_logistic(object = x)
  
  model <- keras_model(inputs = input,outputs = output)
  
  return(model)
}


##
# Composition: VGG-like convnet
vgg_like_convnet = function(input_shape = c(100,100,3),
                            num_classes = 10,
                            filters = c(32,32), #Defines depth; limitation.
                            conv2D_kernel = c(3,3),
                            activation_blocks = "relu",
                            pool_size = c(2,2),
                            dropuout_blocks = 0.25,
                            units = c(256)){ #Defines depth too; limitation.
  # Composition: VGG-like convnet
  # Inspired by: https://keras.rstudio.com/articles/sequential_model.html
  require(keras)
  
  input = layer_input(shape = input_shape)
  
  x <- input
  
  block_length = length(filters)
  for(i in 1:block_length){
    x <- layer_block_2conv2D_maxpool_drop(object = x,
                                          filters_b = filters[i],
                                          kernel_size_b = conv2D_kernel,
                                          activation_b = activation_blocks,
                                          pool_size_b = pool_size,
                                          rate_b = dropuout_blocks)
  }
  
  x <- layer_flatten(object = x)
  
  block_length_p2 = length(units)
  for(i in 1:block_length_p2){
    x <- layer_block_dense_drop(object = x,
                                units_b = units[i],
                                activation_b = activation_blocks,
                                rate_b = dropuout_blocks)
  }
  
  
  output <- layer_end_multinom(object = x,
                              num_classes = num_classes)
  
  model <- keras_model(inputs = input, outputs = output)
  
  return(model)
}






























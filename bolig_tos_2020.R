###
# Boligpriser Tromso Desember 2020

##
# Load libraries
library(keras)
library(tfdatasets)
library(dplyr)
library(ggplot2)

##
# Load data
path_to_data = "boligpriser_tromso_desember_2020.csv"

bpt_data = read.csv(file = path_to_data,sep = ",",header = TRUE)

is.data.frame(bpt_data)

##
# Check data
dim(bpt_data)

n_obs = dim(bpt_data)[1]
m_feat = dim(bpt_data)[2]

names(bpt_data)

apply(X = bpt_data,MARGIN = 2,FUN = function(x) is.factor(x))
apply(X = bpt_data,MARGIN = 2,FUN = function(x) is.numeric(x))
apply(X = bpt_data,MARGIN = 2,FUN = function(x) is.character(x))

sapply(bpt_data, class)

table(bpt_data$bydel_ca,bpt_data$bydel_grov)

hist(bpt_data$tot_pris,breaks = 30)

##
# Create new dataset to correct it
bpt_data_fix = bpt_data

bpt_data_fix$tot_pris = as.numeric(bpt_data_fix$tot_pris)
bpt_data_fix$kvdmeter_P_B = as.numeric(bpt_data_fix$kvdmeter_P_B)

bpt_data_fix$soverom = as.numeric(bpt_data_fix$soverom)
bpt_data_fix$rom_ca = as.numeric(bpt_data_fix$rom_ca)
bpt_data_fix$etasje = as.numeric(bpt_data_fix$etasje)

bpt_data_fix$fellesutgifter = as.numeric(bpt_data_fix$fellesutgifter)

bpt_data_fix$bydel_ca = as.numeric(as.factor(bpt_data_fix$bydel_ca))
bpt_data_fix$bydel_grov = as.numeric(as.factor(bpt_data_fix$bydel_grov))

bpt_data_fix$byggeaar = as.numeric(bpt_data_fix$byggeaar)

bpt_data_fix = bpt_data_fix[,-m_feat]
bpt_data_fix = bpt_data_fix[,-1:-2]

sapply(bpt_data_fix, class)

m_feat_fix = dim(bpt_data_fix)[2]

##
# Train and testing split of data
y_data = bpt_data_fix$tot_pris
x_data = bpt_data_fix[,2:m_feat_fix]

train_to_test_split_ratio = 0.6
n_train = round(train_to_test_split_ratio*n_obs)

# Random selection of training and testing
sampled_indices = as.logical(sample(x = c(rep(0,n_obs - n_train),rep(1,n_train)),size = n_obs, replace = FALSE))
inverse_sampled_indices = !sampled_indices

sum(sampled_indices)
sum(inverse_sampled_indices)

y_train = data.frame(tot_pris = y_data[sampled_indices])
x_train = x_data[sampled_indices,]

train_df = bpt_data_fix[sampled_indices,]

dim(y_train)
dim(x_train)

dim(train_df)

y_test = data.frame(tot_pris = y_data[inverse_sampled_indices]) 
x_test = x_data[inverse_sampled_indices,]

test_df = bpt_data_fix[inverse_sampled_indices,]

dim(y_test)
dim(x_test)

dim(test_df)

##
# Model
names(train_df)

rm(spec)
spec <- feature_spec(train_df, tot_pris ~ .) %>% 
  step_numeric_column(kvdmeter_P_B) %>% 
  step_numeric_column(soverom) %>% 
  step_numeric_column(rom_ca) %>% 
  step_numeric_column(etasje) %>% 
  step_numeric_column(fellesutgifter) %>% 
  step_numeric_column(byggeaar) %>%
  step_categorical_column_with_identity(bydel_ca,num_buckets = length(unique(bpt_data_fix$bydel_ca))) %>%
  step_categorical_column_with_identity(bydel_grov,num_buckets = length(unique(bpt_data_fix$bydel_grov)))

spec <- fit(spec)

rm(layer)
layer <- layer_dense_features(
  feature_columns = dense_features(spec), 
  dtype = tf$float32
)
#layer(train_df)


input <- layer_input_from_dataset(train_df %>% select(-tot_pris))

output <- input %>% 
  layer_dense_features(dense_features(spec)) %>% 
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) 

model <- keras_model(input, output)

summary(model)


early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)

model %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )


history <- model %>% fit(
  x = train_df %>% select(-tot_pris),
  y = train_df$tot_pris,
  epochs = 250,
  validation_split = 0.1,
  callbacks = list(early_stop)
)

plot(history)

test_predictions <- model %>% predict(test_df %>% select(-tot_pris))

pred_true_df = data.frame(pred_y = test_predictions,true_y = test_df$tot_pris)

mean(abs(pred_true_df$pred_y - pred_true_df$true_y))
sd(abs(pred_true_df$pred_y - pred_true_df$true_y))

###
# Model 2 --- more abstract:

layer_block_dense_drop = function(object,
                                  units_b = 32,
                                  activation_b = "relu",
                                  rate_b = 0.1){
  require(keras)
  
  x <- object
  
  x <- layer_dense(object = x, units = units_b,activation = activation_b)
  
  x <- layer_dropout(object = x, rate = rate_b)
  
  return(x)
}


layer_end_continous = function(object,
                               units_b = 16,
                               activation_b = "relu"){
  require(keras)
  
  x <- object
  
  x <- layer_dense(object = x,units = units_b,activation = activation_b)
  
  x <- layer_dense(object = x, units = 1)
  
  return(x)
}

ff_nn_df_regression = function(input_layer = layer_input_from_dataset(train_df %>% select(-tot_pris)),
                               spec_reg = spec,
                               width_blocks = c(32,64,32), #Vector of integers
                               block_dropout = 0.1, #Single
                               block_activation = "relu", #Single
                               end_width = 16,
                               end_activation = "relu"){
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
  
  output <- layer_end_continous(object = x,units_b = end_width,activation_b = end_activation)
  
  model <- keras_model(inputs = input, outputs = output)
  
  return(model)
}

model_2 = ff_nn_df_regression(width_blocks = rep(128,12),
                              block_dropout = 0.2,
                              end_width = 8)

summary(model_2)


model_2 %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_adadelta(),
    metrics = list("mean_squared_error")
  )


history <- model_2 %>% fit(
  x = train_df %>% select(-tot_pris),
  y = train_df$tot_pris,
  epochs = 250,
  validation_split = 0.3,
  callbacks = list(early_stop)
)

#plot(history)

test_predictions <- model_2 %>% predict(test_df %>% select(-tot_pris))

pred_true_df = data.frame(pred_y = test_predictions,true_y = test_df$tot_pris)

mean(abs(pred_true_df$pred_y - pred_true_df$true_y))
sd(abs(pred_true_df$pred_y - pred_true_df$true_y))

### Testing block models.

model_bin <- mlp_bin_cat(input_shape = c(20))

summary(model_bin)

model_mult <- mlp_mult_cat(input_shape = c(20),num_classes = 4)

summary(model_mult)

model_vgg <- vgg_like_convnet(input_shape = c(100,100,3),
                              num_classes = 3)

summary(model_vgg)


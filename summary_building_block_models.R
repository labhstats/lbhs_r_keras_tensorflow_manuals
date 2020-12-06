library(keras)

#"Testing" models of building blocks


#MLP sigmoid
model_bin <- mlp_bin_cat(input_shape = c(20))
summary(model_bin)

#MLP softmax
model_mult <- mlp_mult_cat(input_shape = c(20),
                           num_classes = 4)
summary(model_mult)

#VGG ish(?)
model_vgg <- vgg_like_convnet(input_shape = c(100,100,3),
                              num_classes = 3)
summary(model_vgg)





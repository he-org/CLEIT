from tensorflow import keras
#encoder configurations
encoder_architecture = [512, 256, 128]
encoder_latent_dimension = 128
encoder_act_fn = keras.activations.relu
encoder_output_act_fn = None

#regressor module configurations
regressor_architecture = [128, 128, 64, 16]
regressor_shared_layer_number = 2
regressor_act_fn = keras.activations.relu
regressor_output_dim = 265
regressor_output_act_fn =keras.activations.sigmoid
#transmitter module configurations
transmitter_architecture = [128, 128]
transmitter_act_fn = keras.activations.relu
transmitter_output_act_fn = None
transmitter_output_dim = encoder_latent_dimension
#learning configurations
kernel_regularizer_l = 0.001
pre_training_lr = 1e-3
fine_tuning_lr = 5e-4
decay = 0.8
max_epoch = 400
min_epoch = 100
gradual_unfreezing_flag = True
unfrozen_epoch = 40
batch_size = 64
alpha = 1.0
gradient_threshold = None


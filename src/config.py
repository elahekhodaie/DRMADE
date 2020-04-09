import torch
from torchvision import datasets
from pathlib import Path
from torch.distributions.normal import Normal
import torch as t

# model architecture
latent_size = 16

made_hidden_layers = [64, 64, 64]
made_num_masks = 16
made_use_biases = True
made_natural_ordering = True

num_mix = 5
distribution = Normal
num_dist_parameters = 2
parameters_transform = [lambda x: x, t.exp]  # mean, std
paramteres_min_value = [0.0, 0.5]
parameters_regularization = [lambda x: 0, lambda x: t.sum(1 / x)]

encoder_use_bias = False
encoder_bn_affine = False
encoder_bn_eps = 1e-4
encoder_bn_latent = False
encoder_num_layers = 2
encoder_latent_activation = ''  # '', tanh, leaky_relu
encoder_layers_activation = 'elu'  # leaky_relu, elu, relu

decoder_use_bias = False
decoder_bn_affine = False
decoder_bn_eps = 1e-4
decoder_num_layers = 2
decoder_output_activation = 'tanh'  # '', tanh, sigmoid
decoder_layers_activation = 'elu'  # leaky_relu, elu, relu
decoder_distance_norm = 2

# training
train_dataset = datasets.MNIST
normal_classes = [8]

test_classes = None
test_dataset = datasets.MNIST

lr_decay = 0.999995  # Learning rate decay, applied every step of the optimization
lr_half_schedule = 512  # interval of epochs to reduce learning rate 50%
base_lr = 0.0002

noising_factor = 0  # the noise to add to each input while training the model
latent_cor_regularization_factor = 0.01
latent_cor_regularization_abs = False
latent_zero_regularization_factor = 0.01
latent_zero_regularization_eps = 1e-3
latent_variance_regularization_factor = 0.1
latent_distance_regularization_factor = 0.1
distance_factor = 1
latent_distance_normalize_features = False
latent_distance_norm = 2
parameters_regularization_factor = [0, 1]

max_epoch = 2050

# evaluation
positive_is_anomaly = False

# Reproducability
seed = 1  # Random seed to use

# data
# data loader
train_batch_size = 512
validation_batch_size = 128
test_batch_size = 512
dataloader_num_workers = 4
dataloader_pin_memory = True
dataloader_shuffle = True
dataloader_drop_last = True

# data I/O
save_interval = 16
validation_interval = 16
evaluation_interval = 16
commit_images_interval = 4
embedding_interval = 1024
track_weights_interval = 64
log_data_feed_loop_interval = 8
log_evaluation_loop_interval = 8
num_extreme_cases = 16

output_root = './output'
data_dir = '.' + '/data'  # Location for the dataset
models_dir = output_root + '/models'  # Location for parameter checkpoints and samples
runs_dir = output_root + '/runs'
log_dir = output_root + '/log'


def set_output_dirs(root='./output'):
    global output_root, data_dir, models_dir, runs_dir, log_dir
    output_root = root
    data_dir = '.' + '/data'  # Location for the dataset
    models_dir = output_root + '/models'  # Location for parameter checkpoints and samples
    runs_dir = output_root + '/runs'
    log_dir = output_root + '/log'
    # losses_dir = log_dir + '/losses'
    # evaluation_dir = log_dir + '/evaluation'
    # extreme_cases_dir = log_dir + '/extreme_cases'

    # ensuring the existance of output directories
    Path(output_root).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # Path(evaluation_dir).mkdir(parents=True, exist_ok=True)
    # Path(losses_dir).mkdir(parents=True, exist_ok=True)
    # Path(extreme_cases_dir).mkdir(parents=True, exist_ok=True)

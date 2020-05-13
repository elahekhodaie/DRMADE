import torch
from torchvision import datasets
from pathlib import Path
from torch.distributions.normal import Normal
import torch as t

# model architecture
latent_size = 16

made_hidden_layers = [132, 132, 132]
made_num_masks = 5
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
decoder_output_activation = 'sigmoid'  # '', tanh, sigmoid
decoder_layers_activation = 'elu'  # leaky_relu, elu, relu
decoder_distance_norm = 2
decoder_distance_eps = 1e-5

# training
input_limits = (-1., 1.)
dataset = datasets.CIFAR10
normal_classes = [1]
input_rescaling = lambda x: (x - .5) * 2.
input_rescaling_inv = lambda x: x * .5 + .5

lr_decay = 0.999995  # Learning rate decay, applied every step of the optimization
lr_half_schedule = 512  # interval of epochs to reduce learning rate 50%
base_lr = 0.0002

noising_factor = 0  # the noise to add to each input while training the model
latent_cor_regularization_factor = 0.
latent_cor_regularization_abs = False
latent_zero_regularization_factor = 0.
latent_zero_regularization_eps = 1e-3
latent_variance_regularization_factor = 0.
latent_distance_regularization_factor = 0.
distance_factor = 1
latent_distance_normalize_features = False
latent_distance_norm = 2
parameters_regularization_factor = [0, 1]

pretrain_ae_pgd_eps = 0.2
pretrain_ae_pgd_iterations = 20
pretrain_ae_pgd_alpha = 0.05
pretrain_ae_pgd_randomize = False

pretrain_ae_latent_pgd_eps = 0
pretrain_ae_latent_pgd_iterations = 20
pretrain_ae_latent_pgd_alpha = 0.05
pretrain_ae_latent_pgd_randomize = False

pretrain_made_pgd_eps = 0
pretrain_made_pgd_iterations = 1
pretrain_made_pgd_alpha = 0.05
pretrain_made_pgd_randomize = False

pretrain_encoder_made_pgd_eps = 0.2
pretrain_encoder_made_pgd_iterations = 1
pretrain_encoder_made_pgd_alpha = 0.05
pretrain_encoder_made_pgd_randomize = False

max_epoch = 2050

checkpoint_drmade = None
checkpoint_encoder = None
checkpoint_decoder = None
checkpoint_made = None
# evaluation
positive_is_anomaly = False

# Reproducability
seed = 1  # Random seed to use

# data
# data loader
train_batch_size = 256
validation_batch_size = 256
test_batch_size = 256
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

# deepsvdd
deepsvdd_lambda = 0.5
deepsvdd_temperature = 0.07
deepsvdd_k = 200
deepsvdd_latent_size = 32

deepsvdd_sgd_base_lr = 3e-2
deepsvdd_sgd_lr_decay = 0.999995
deepsvdd_sgd_schedule = 40

deepsvdd_pgd_eps = 0
deepsvdd_pgd_iterations = 20
deepsvdd_pgd_alpha = 0.05
deepsvdd_pgd_randomize = False

output_root = './output'
data_dir = '.' + '/data'  # Location for the dataset
Path(data_dir).mkdir(parents=True, exist_ok=True)

import torch
from torchvision import datasets
from pathlib import Path
from torch.distributions.normal import Normal
import torch as t

# model architecture
latent_size = 16

num_mix = 5
distribution = Normal
num_dist_parameters = 2
parameters_transform = [lambda x: x, t.abs]  # mean, std

made_hidden_layers = [24, 24]
made_use_biases = True
made_num_masks = 16
made_natural_ordering = True

encoder_use_bias = False
encoder_bn_affine = False
encoder_bn_eps = 1e-4

# training
train_dataset = datasets.MNIST
normal_classes = [8]

test_classes = None
test_dataset = datasets.MNIST

lr_decay = 0.999995  # Learning rate decay, applied every step of the optimization
lr_half_schedule = 400  # interval of epochs to reduce learning rate 50%
base_lr = 0.0002

noising_factor = 0.1  # the noise to add to each input while training the model
latent_regularization_factor = 1

max_epoch = 5000

# evaluation
positive_is_anomaly = False

# Reproducability
seed = 1  # Random seed to use

# data
# data loader
batch_size = 64  # Batch size during training per GPU
test_batch_size = batch_size
dataloader_num_workers = 4
dataloader_pin_memory = True
dataloader_shuffle = True
dataloader_drop_last = True

# data I/O
save_interval = 16
log_train_loop_interval = 1
log_validation_loop_interval = 1
log_evaluation_loop_interval = 1

output_root = './output'
data_dir = '.' + '/data'  # Location for the dataset
models_dir = output_root + '/models'  # Location for parameter checkpoints and samples
runs_dir = output_root + '/runs'
log_dir = output_root + '/log'
losses_dir = log_dir + '/losses'
evaluation_dir = log_dir + '/evaluation'
extreme_cases_dir = log_dir + '/extreme_cases'

# ensuring the existance of output directories
Path(output_root).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(evaluation_dir).mkdir(parents=True, exist_ok=True)
Path(losses_dir).mkdir(parents=True, exist_ok=True)
Path(extreme_cases_dir).mkdir(parents=True, exist_ok=True)

import torch
from torchvision import datasets
from pathlib import Path

# model architecture
latent_size = 16
num_dist_parameters = 2

made_hidden_layers = [24, 24]
made_use_biases = True
made_num_masks = 16
made_natural_ordering = True

encoder_use_bias = False
encoder_bn_affine = False
encoder_bn_eps = 1e-4

train_sample_num_masks = 16
test_sample_num_masks = 16

# training
train_dataset = datasets.MNIST
normal_classes = [8]

test_classes = list(range(0, 10))
test_dataset = datasets.MNIST
# data

# data loader
batch_size = 64  # Batch size during training per GPU
test_batch_size = batch_size
dataloader_num_workers = 4
dataloader_pin_memory = True
dataloader_shuffle = True
dataloader_drop_last = True

# data I/O
output_root = '.'
data_dir = output_root + '/data'  # Location for the dataset
models_dir = output_root + '/models'  # Location for parameter checkpoints and samples
log_dir = output_root + '/log'
samples_dir = log_dir + '/samples'
losses_dir = log_dir + '/losses'
evaluation_dir = log_dir + '/evaluation'
extreme_cases_dir = log_dir + '/extreme_cases'

# ensuring the existance of output directories
Path(output_root).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(models_dir).mkdir(parents=True, exist_ok=True)
Path(samples_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(evaluation_dir).mkdir(parents=True, exist_ok=True)
Path(losses_dir).mkdir(parents=True, exist_ok=True)
Path(extreme_cases_dir).mkdir(parents=True, exist_ok=True)

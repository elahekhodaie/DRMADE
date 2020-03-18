import time
import torch as t
from torch.optim import lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import numpy as np

from sklearn.metrics import roc_auc_score
from src.utils.data import DatasetSelection
from src.model.model import DRMADE
import src.config as config

np.seterr(divide='ignore', invalid='ignore')

# variables
batch_size = config.batch_size
test_batch_size = config.test_batch_size

device = t.device("cuda" if t.cuda.is_available() else "cpu")

hidden_layers = config.made_hidden_layers
num_masks = config.made_num_masks
num_mix = config.num_mix
latent_size = config.latent_size

latent_regularization_factor = config.latent_regularization_factor
noise_factor = config.noising_factor

base_lr = config.base_lr
lr_decay = config.lr_decay
lr_half_schedule = config.lr_half_schedule

# accessory functions
noise_function = lambda x: noise_factor * (
        2 * t.FloatTensor(*x).to(device).uniform_() - 1)  # (x will be the input shape tuple)
lr_multiplicative_factor_lambda = lambda epoch: 0.5 if (epoch + 1) % lr_half_schedule == 0 else lr_decay

# reproducibility
t.manual_seed(config.seed)
np.random.seed(config.seed)

# type initialization
t.set_default_tensor_type('torch.FloatTensor')

print('loading training data')
train_data = DatasetSelection(datasets.MNIST, classes=config.normal_classes, train=True)
print('loading validation data')
validation_data = DatasetSelection(datasets.MNIST, classes=config.normal_classes, train=False)
print('loading test data')
test_data = DatasetSelection(datasets.MNIST, train=False)

input_shape = train_data.input_shape()

print('initializing data loaders')
train_loader = train_data.get_dataloader(shuffle=True)
validation_loader = validation_data.get_dataloader(shuffle=False)
test_loader = test_data.get_dataloader(shuffle=False, )

print('initializing model')
model = DRMADE(input_shape[0], latent_size, hidden_layers, num_masks=num_masks, num_mix=num_mix)

# setting up tensorboard data summerizer
model_name = f'{model.name}-rl={latent_regularization_factor}-nz={noise_factor}-Adam,lr={base_lr},dc={lr_decay},s={lr_half_schedule}'
writer = SummaryWriter(
    log_dir=config.runs_dir + f'/{model_name}')

print('initializing optimizer')
optimizer = Adam(model.parameters(), lr=base_lr)
print('initializing learning rate scheduler')
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_multiplicative_factor_lambda,
                                          last_epoch=-1)


def train_loop(epoch):
    train_loss = 0.0
    time_ = time.time()
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        if noise_factor:
            noisy_images = images + noise_function(images.shape)
            noisy_images.clamp_(min=-1, max=1)

        loss = 0.0
        for i in range(num_masks):
            model.made.update_masks()
            if noise_factor:
                output, noised_features = model(noisy_images)
                features = model.encoder(images)
            else:
                output, features = model(images)
            loss += -model.log_prob(features, output) + latent_regularization_factor * model.latent_regularization_term(
                features)
        loss /= num_masks
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss / batch_size
        if config.log_train_loop_interval and (batch_idx + 1) % config.log_train_loop_interval == 0:
            print(
                '\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                    batch_idx, len(train_loader), train_loss / (1 + batch_idx), time.time() - time_)
            )
            time_ = time.time()
    return train_loss / len(train_loader)


def validation_loop(epoch):
    validation_loss = 0.0
    time_ = time()
    with t.no_grad():
        for batch_idx, (images, _) in enumerate(validation_loader):
            images = images.to(device)
            loss = 0.0
            for i in range(num_masks):
                model.made.update_masks()
                output, features = model(images)
                loss += -model.log_prob(features,
                                        output) + latent_regularization_factor * model.latent_regularization_term(
                    features)
            loss /= num_masks
            validation_loss += loss / batch_size
            if config.log_validation_loop_interval and (batch_idx + 1) % config.log_validation_loop_interval == 0:
                print(
                    '\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                        batch_idx, len(validation_loader), validation_loss / (1 + batch_idx), time.time() - time_)
                )
                time_ = time.time()
    return validation_loss / len(validation_loader)


def evaluate_loop(epoch):
    with t.no_grad():
        scores = t.Tensor().to(device)
        labels = np.empty(0, dtype=np.int8)
        for batch_idx, (images, label) in enumerate(test_loader):
            images = images.to(device)
            output, features = model(images)
            scores = t.cat((scores, model.log_prob_hitmap(features, output).sum(1)), dim=0)
            labels = np.append(labels, label.numpy().astype(np.int8), axis=0)
        is_pos = np.isin(labels, [8])
    return scores, is_pos


def train():
    for epoch in range(config.max_epoch):
        print('epoch {:4} - lr: {}'.format(epoch, optimizer.param_groups[0]["lr"]))
        validation_loss = validation_loop(epoch)
        scores, is_pos = evaluate_loop(epoch)
        train_loss = train_loop(epoch)

        writer.add_scalars('loss', {'validation': validation_loss, 'training': train_loss}, epoch)
        writer.add_scalar('auc', roc_auc_score(y_true=is_pos.astype(np.int8), y_score=scores), epoch)
        writer.add_histogram('test/positive-scores', scores[is_pos], epoch)
        writer.add_histogram('test/negative-scores', scores[(is_pos == False)], epoch)

        if config.save_interval and (epoch + 1) % config.save_interval == 0:
            model.save(config.models_dir + f'/{model_name}-E{epoch}.pth')

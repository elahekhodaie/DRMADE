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

latent_cor_regularization_factor = config.latent_cor_regularization_factor
latent_zero_regularization_factor = config.latent_zero_regularization_factor

noise_factor = config.noising_factor

base_lr = config.base_lr
lr_decay = config.lr_decay
lr_half_schedule = config.lr_half_schedule

# accessory functions
noise_function = lambda x: noise_factor * (
        2 * t.DoubleTensor(*x).to(device).uniform_() - 1)  # (x will be the input shape tuple)
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
train_loader = train_data.get_dataloader(shuffle=True, batch_size=batch_size)
validation_loader = validation_data.get_dataloader(shuffle=False, batch_size=batch_size)
test_loader = test_data.get_dataloader(shuffle=False, batch_size=test_batch_size)

print('initializing model')
model = DRMADE(input_shape[0], latent_size, hidden_layers, num_masks=num_masks, num_mix=num_mix).to(device)
model.encoder = model.encoder.to(device)
model.made = model.made.to(device)

# setting up tensorboard data summerizer
model_name = '{}-rl_cor={}{}-rl_zero={}eps{}-nz={}-Adam,lr={},dc={},s={}'.format(
    model.name,
    "abs" if config.latent_cor_regularization_abs else "noabs",
    latent_cor_regularization_factor,
    latent_zero_regularization_factor,
    config.latent_zero_regularization_eps,
    noise_factor,
    base_lr,
    lr_decay,
    lr_half_schedule
)
writer = SummaryWriter(
    log_dir=config.runs_dir + f'/{model_name}')

print('initializing optimizer')
optimizer = Adam(model.parameters(), lr=base_lr)
print('initializing learning rate scheduler')
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_multiplicative_factor_lambda,
                                          last_epoch=-1)


def train_loop():
    data = {
        'loss': 0.0,
        'log_prob': 0.0,
        'latent_regularization/correlation': 0.0,
        'latent_regularization/zero': 0.0,
        'parameters_regularization': [0.0 for i in range(config.num_dist_parameters)]
    }

    time_ = time.time()
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        if noise_factor:
            noisy_images = images + noise_function(images.shape)
            noisy_images.clamp_(min=-1, max=1)
            noised_features = model.encoder(noisy_images)
        features = model.encoder(images)

        log_prob = 0.0
        parameters_regularization = [0.0 for i in range(config.num_dist_parameters)]
        for i in range(num_masks):
            model.made.update_masks()
            if noise_factor:
                noised_output = model.made(noised_features)
                parameters = model.get_dist_parameters(noised_output)
                log_prob += model.log_prob(features, noised_output, parameters=parameters)
            else:
                output = model.made(features)
                parameters = model.get_dist_parameters(output)
                log_prob += model.log_prob(features, output=output, parameters=parameters)
            for j, regularization in enumerate(config.parameters_regularization):
                parameters_regularization[j] += regularization(parameters[j])

        log_prob /= num_masks
        for i in range(config.num_dist_parameters):
            parameters_regularization[i] /= num_masks

        latent_cor_regularization = model.latent_cor_regularization(
            noised_features) if noise_factor else model.latent_cor_regularization(features)

        latent_zero_regularization = model.latent_zero_regularization(
            noised_features) if noise_factor else model.latent_zero_regularization(features)

        loss = -log_prob + latent_cor_regularization_factor * latent_cor_regularization + latent_zero_regularization_factor * latent_zero_regularization
        for i, factor in enumerate(config.parameters_regularization_factor):
            loss += factor * parameters_regularization[i]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        data['loss'] += loss / batch_size
        data['log_prob'] += log_prob / batch_size
        data['latent_regularization/correlation'] += latent_cor_regularization / batch_size
        data['latent_regularization/zero'] += latent_zero_regularization / batch_size
        for i, reg in enumerate(parameters_regularization):
            data['parameters_regularization'][i] += reg / num_masks

        if config.log_train_loop_interval and (batch_idx + 1) % config.log_train_loop_interval == 0:
            print(
                '\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                    batch_idx, len(train_loader), data['loss'] / (1 + batch_idx), time.time() - time_)
            )
            time_ = time.time()
        for key in data:
            if isinstance(data[key], list):
                for i in range(len(data[key])):
                    data[key][i] /= len(train_loader)
            else:
                data[key] /= len(train_loader)
    return data


def validation_loop():
    data = {
        'loss': 0.0,
        'log_prob': 0.0,
        'latent_regularization/correlation': 0.0,
        'latent_regularization/zero': 0.0,
        'parameters_regularization': [0.0 for i in range(config.num_dist_parameters)]
    }

    time_ = time.time()
    with t.no_grad():
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            if noise_factor:
                noisy_images = images + noise_function(images.shape)
                noisy_images.clamp_(min=-1, max=1)
                noised_features = model.encoder(noisy_images)
            features = model.encoder(images)

            log_prob = 0.0
            parameters_regularization = [0.0 for i in range(config.num_dist_parameters)]
            for i in range(num_masks):
                model.made.update_masks()
                output = model.made(features)
                parameters = model.get_dist_parameters(output)
                log_prob += model.log_prob(features, output=output, parameters=parameters)
                for j, regularization in enumerate(config.parameters_regularization):
                    parameters_regularization[j] += regularization(parameters[j])

            log_prob /= num_masks
            for i in range(config.num_dist_parameters):
                parameters_regularization[i] /= num_masks

            latent_cor_regularization = model.latent_cor_regularization(
                noised_features) if noise_factor else model.latent_cor_regularization(features)

            latent_zero_regularization = model.latent_zero_regularization(
                noised_features) if noise_factor else model.latent_zero_regularization(features)

            loss = -log_prob + latent_cor_regularization_factor * latent_cor_regularization + \
                   latent_zero_regularization_factor * latent_zero_regularization
            for i, factor in enumerate(config.parameters_regularization_factor):
                loss += factor * parameters_regularization[i]

            data['loss'] += loss / batch_size
            data['log_prob'] += log_prob / batch_size
            data['latent_regularization/correlation'] += latent_cor_regularization / batch_size
            data['latent_regularization/zero'] += latent_zero_regularization / batch_size
            for i, reg in enumerate(parameters_regularization):
                data['parameters_regularization'][i] += reg / num_masks

            if config.log_validation_loop_interval and (batch_idx + 1) % config.log_validation_loop_interval == 0:
                print(
                    '\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                        batch_idx, len(train_loader), data['loss'] / (1 + batch_idx), time.time() - time_)
                )
                time_ = time.time()
        for key in data:
            if isinstance(data[key], list):
                for i in range(len(data[key])):
                    data[key][i] /= len(train_loader)
            else:
                data[key] /= len(train_loader)
    return data


def evaluate_loop(data_loader, record_input_images=False):
    with t.no_grad():
        scores = t.Tensor().to(device)
        features = t.Tensor().to(device)
        labels = np.empty(0, dtype=np.int8)
        input_images = t.Tensor().to(device)
        time_ = time.time()
        for batch_idx, (images, label) in enumerate(data_loader):
            images = images.to(device)
            if record_input_images:
                input_images = t.cat((input_images, images), dim=0)

            output, latent = model(images)
            scores = t.cat((scores, model.log_prob_hitmap(latent, output).sum(1)), dim=0)
            features = t.cat((features, latent), dim=0)
            labels = np.append(labels, label.numpy().astype(np.int8), axis=0)
            if config.log_evaluation_loop_interval and (batch_idx + 1) % config.log_evaluation_loop_interval == 0:
                print(
                    '\t{:3d}/{:3d} - time : {:.3f}s'.format(
                        batch_idx, len(data_loader), time.time() - time_)
                )
                time_ = time.time()
    return scores, features, labels, input_images


def submit_loop_data(data, title, epoch):
    for key in data:
        if isinstance(data[key], list):
            for i in range(len(data[key])):
                writer.add_scalar(f'{key}/{title}/{i}', data[key][i], epoch)
        else:
            writer.add_scalar(f'{key}/{title}', data[key], epoch)


def submit_encoder_weights(epoch):
    writer.add_histogram('encoder/conv1', model.encoder.conv1.weight, epoch)
    writer.add_histogram('encoder/conv2', model.encoder.conv2.weight, epoch)
    writer.add_histogram('encoder/conv3', model.encoder.conv3.weight, epoch)
    writer.add_histogram('encoder/fc1', model.encoder.fc1.weight, epoch)


def train():
    for epoch in range(5121):
        print('epoch {:4} - lr: {}'.format(epoch, optimizer.param_groups[0]["lr"]))
        if config.validation_interval and epoch % config.validation_interval == 0:
            validation_results = validation_loop()
            submit_loop_data(validation_results, 'validation', epoch)

        if config.evaluation_interval and epoch % config.evaluation_interval == 0:
            scores, features, labels, _ = evaluate_loop(test_loader)
            writer.add_scalar('auc', roc_auc_score(y_true=np.isin(labels, config.normal_classes).astype(np.int8),
                                                   y_score=scores.cpu()), epoch)
            writer.add_histogram('log_probs/anomaly', scores[(np.isin(labels, config.normal_classes) == False)], epoch)
            scores, features, labels, _ = evaluate_loop(train_loader)
            writer.add_histogram('log_probs/normal', scores, epoch)
            writer.flush()

        if config.embedding_interval and (epoch + 1) % config.embedding_interval == 0:
            scores, features, labels, input_images = evaluate_loop(test_loader, record_input_images=True)
            writer.add_embedding(features, metadata=labels, label_img=input_images, global_step=epoch,
                                 tag=f'{model_name}/test')
            writer.flush()

        if config.track_weights_interval and epoch % config.track_weights_interval == 0:
            submit_encoder_weights(epoch)

        train_results = train_loop()
        submit_loop_data(train_results, 'train', epoch)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()

        if config.save_interval and (epoch + 1) % config.save_interval == 0:
            model.save(config.models_dir + f'/{model_name}-E{epoch}.pth')

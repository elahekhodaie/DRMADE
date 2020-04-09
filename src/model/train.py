import time
import torch as t
from torch.optim import lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from pathlib import Path
import numpy as np

from sklearn.metrics import roc_auc_score
from src.utils.data import DatasetSelection
from src.model.model import DRMADE
import src.config as config

np.seterr(divide='ignore', invalid='ignore')

config.set_output_dirs('./output')

# variables
train_batch_size = 512
validation_batch_size = 128
test_batch_size = 512

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print("device:", device)

hidden_layers = config.made_hidden_layers
num_masks = config.made_num_masks
num_mix = config.num_mix
latent_size = config.latent_size

latent_cor_regularization_factor = config.latent_cor_regularization_factor
latent_zero_regularization_factor = config.latent_zero_regularization_factor
latent_variance_regularization_factor = config.latent_variance_regularization_factor
latent_distance_regularization_factor = config.latent_distance_regularization_factor
distance_factor = config.distance_factor

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
train_loader = train_data.get_dataloader(shuffle=True, batch_size=train_batch_size)
validation_loader = validation_data.get_dataloader(shuffle=False, batch_size=validation_batch_size)
test_loader = test_data.get_dataloader(shuffle=False, batch_size=test_batch_size)

print('initializing model')
model = DRMADE(input_shape[1], input_shape[0], latent_size, hidden_layers, num_masks=num_masks, num_mix=num_mix).to(
    device)
model.encoder = model.encoder.to(device)
model.made = model.made.to(device)
model.decoder = model.decoder.to(device)

# setting up tensorboard data summerizer
model_name = '{}{}{}{}{}{}|distance{}|Adam,lr{},dc{},s{}'.format(
    model.name,
    '|rl_correlation{}{}'.format(
        "abs" if config.latent_cor_regularization_abs else "noabs",
        latent_cor_regularization_factor,
    ) if latent_cor_regularization_factor else '',
    '|rl_variance{}'.format(
        latent_variance_regularization_factor,
    ) if latent_variance_regularization_factor else '',
    '|rl_zero{}eps{}'.format(
        latent_zero_regularization_factor,
        config.latent_zero_regularization_eps,
    ) if latent_zero_regularization_factor else '',
    '|rl_distance{}{},norm{}'.format(
        'normalized' if config.latent_distance_normalize_features else '',
        latent_distance_regularization_factor,
        config.latent_distance_norm,
    ) if latent_distance_regularization_factor else '',
    '|nz{}'.format(noise_factor) if noise_factor else '',
    distance_factor,
    base_lr,
    lr_decay,
    lr_half_schedule
)
print(model_name)
Path(config.models_dir + f'/{model_name}').mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(
    log_dir=config.runs_dir + f'/{model_name}')

print('initializing optimizer')
optimizer = Adam(model.parameters(), lr=base_lr)
print('initializing learning rate scheduler')
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_multiplicative_factor_lambda,
                                          last_epoch=-1)


def data_feed_loop(data_loader, optimize=True, log_interval=config.log_data_feed_loop_interval, loop_name=''):
    data = {
        'loss': 0.0,
        'log_prob': 0.0,
        'decoder_distance': 0.0,
        'latent_regularization/correlation': 0.0,
        'latent_regularization/zero': 0.0,
        'latent_regularization/variance': 0.0,
        'latent_regularization/distance': 0.0,
        'parameters_regularization': [0.0 for i in range(config.num_dist_parameters)]
    }

    time_ = time.time()
    with t.set_grad_enabled(optimize):
        for batch_idx, (images, _) in enumerate(data_loader):
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
                    parameters = model.made.get_dist_parameters(noised_output)
                    log_prob += model.made.log_prob(features, noised_output, parameters=parameters)
                else:
                    output = model.made(features)
                    parameters = model.made.get_dist_parameters(output)
                    log_prob += model.made.log_prob(features, output=output, parameters=parameters)
                for j, regularization in enumerate(config.parameters_regularization):
                    parameters_regularization[j] += regularization(parameters[j])

            log_prob /= num_masks

            for i in range(config.num_dist_parameters):
                parameters_regularization[i] /= num_masks

            latent_cor_regularization = model.encoder.latent_cor_regularization(
                noised_features) if noise_factor else model.encoder.latent_cor_regularization(features)
            latent_var_regularization = model.encoder.latent_var_regularization(
                noised_features) if noise_factor else model.encoder.latent_var_regularization(features)
            latent_zero_regularization = model.encoder.latent_zero_regularization(
                noised_features) if noise_factor else model.encoder.latent_zero_regularization(features)
            latent_distance_regularization = model.encoder.latent_distance_regularization(
                noised_features) if noise_factor else model.encoder.latent_distance_regularization(features)

            reconstructed_image = model.decoder(noised_features) if noise_factor else model.decoder(features)
            decoder_distance = model.decoder.distance(images, reconstructed_image).sum()

            loss = -log_prob + distance_factor * decoder_distance
            if latent_cor_regularization_factor:
                loss += latent_cor_regularization_factor * latent_cor_regularization
            if latent_zero_regularization_factor:
                loss += latent_zero_regularization_factor * latent_zero_regularization
            if latent_variance_regularization_factor:
                loss += -latent_variance_regularization_factor * latent_var_regularization
            if latent_distance_regularization_factor:
                loss += latent_distance_regularization_factor * latent_distance_regularization

            for i, factor in enumerate(config.parameters_regularization_factor):
                if factor:
                    loss += factor * parameters_regularization[i]

            data['loss'] += loss / images.shape[0]
            data['log_prob'] += log_prob / images.shape[0]
            data['decoder_distance'] += decoder_distance / images.shape[0]
            data['latent_regularization/correlation'] += latent_cor_regularization / images.shape[0]
            data['latent_regularization/zero'] += latent_zero_regularization / images.shape[0]
            data['latent_regularization/variance'] += latent_var_regularization / images.shape[0]
            data['latent_regularization/distance'] += latent_distance_regularization / images.shape[0]
            for i, reg in enumerate(parameters_regularization):
                data['parameters_regularization'][i] += reg / num_masks

            if optimize:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if log_interval and (batch_idx + 1) % log_interval == 0:
                print(
                    '\t{}\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                        loop_name, batch_idx, len(data_loader), data['loss'] / (1 + batch_idx), time.time() - time_)
                )
                time_ = time.time()
    for key in data:
        if isinstance(data[key], list):
            for i in range(len(data[key])):
                data[key][i] /= len(data_loader)
        else:
            data[key] /= len(data_loader)
    return data


def evaluate_loop(data_loader, record_input_images=False, record_reconstructions=False, loop_name='evaluation'):
    with t.no_grad():
        scores = t.Tensor().to(device)
        reconstructed_images = t.Tensor().to(device)
        features = t.Tensor().to(device)
        labels = np.empty(0, dtype=np.int8)
        input_images = t.Tensor().to(device)
        time_ = time.time()
        for batch_idx, (images, label) in enumerate(data_loader):
            images = images.to(device)
            if record_input_images:
                input_images = t.cat((input_images, images), dim=0)

            output, latent, reconstruction = model(images)
            scores = t.cat((scores, model.made.log_prob_hitmap(latent, output).sum(1)), dim=0)
            features = t.cat((features, latent), dim=0)
            if record_reconstructions:
                reconstructed_images = t.cat((reconstructed_images, reconstruction), dim=0)
            labels = np.append(labels, label.numpy().astype(np.int8), axis=0)
            if config.log_evaluation_loop_interval and (batch_idx + 1) % config.log_evaluation_loop_interval == 0:
                print(
                    '\t{}\t{:3d}/{:3d} - time : {:.3f}s'.format(
                        loop_name, batch_idx, len(data_loader), time.time() - time_)
                )
                time_ = time.time()
    return scores, features, labels, input_images, reconstructed_images


def submit_extreme_cases(input_images, reconstructed_images, epoch, tag='worst_reconstructions',
                         num_cases=config.num_extreme_cases):
    distance_hitmap = model.decoder.distance_hitmap(input_images, reconstructed_images).detach().cpu().numpy()
    distance = model.decoder.distance(input_images, reconstructed_images).detach().cpu().numpy()
    sorted_indexes = np.argsort(distance)
    result_images = np.empty((num_cases * 3, input_images.shape[1], input_images.shape[2], input_images.shape[3]))
    for i, index in enumerate(sorted_indexes[:num_cases]):
        result_images[i * 3] = input_images[index]
        result_images[i * 3 + 1] = reconstructed_images[index]
        result_images[i * 3 + 2] = distance_hitmap[index]
    writer.add_images(tag, result_images, epoch)


def submit_loop_data(data, title, epoch):
    for key in data:
        if isinstance(data[key], list):
            for i in range(len(data[key])):
                writer.add_scalar(f'{key}/{title}/{i}', data[key][i], epoch)
        else:
            writer.add_scalar(f'{key}/{title}', data[key], epoch)


def submit_network_weights(epoch):
    for i, conv in enumerate(model.encoder.conv_layers):
        writer.add_histogram(f'encoder/conv{i}', conv.weight, epoch)
    for i, deconv in enumerate(model.decoder.deconv_layers):
        writer.add_histogram(f'decoder/deconv{i}', deconv.weight, epoch)


def submit_features(features, tag, epoch):
    vector_size = features.shape[1]
    for i in range(vector_size):
        writer.add_histogram(f'{tag}/{i}', features[:, i], epoch)


def train():
    for epoch in range(config.max_epoch):
        print('epoch {:4} - lr: {}'.format(epoch, optimizer.param_groups[0]["lr"]))
        if config.validation_interval and epoch % config.validation_interval == 0:
            validation_results = data_feed_loop(validation_loader, False, loop_name='validation')
            submit_loop_data(validation_results, 'validation', epoch)

        if config.evaluation_interval and epoch % config.evaluation_interval == 0:
            record_extreme_cases = config.commit_images_interval and (epoch % (
                    config.evaluation_interval * config.commit_images_interval) == 0)
            scores, features, labels, images, reconstruction = evaluate_loop(
                test_loader, record_extreme_cases, record_extreme_cases, loop_name='test evaluation')
            writer.add_scalar(
                f'auc{str(config.normal_classes)}',
                roc_auc_score(y_true=np.isin(labels, config.normal_classes).astype(np.int8),
                              y_score=scores.cpu()), epoch)
            anomaly_indexes = (np.isin(labels, config.normal_classes) == False)
            writer.add_histogram('log_probs/test/anomaly', scores[anomaly_indexes], epoch)
            writer.add_histogram('log_probs/test/normal', scores[(anomaly_indexes == False)], epoch)
            if record_extreme_cases:
                submit_extreme_cases(
                    images[anomaly_indexes], reconstruction[anomaly_indexes], epoch,
                    tag='worst_reconstruction/test/anomaly')
                submit_extreme_cases(
                    images[(anomaly_indexes == False)], reconstruction[(anomaly_indexes == False)], epoch,
                    tag='worst_reconstruction/test/normal')
            submit_features(features[anomaly_indexes], 'features/test/anomaly', epoch)
            submit_features(features[(anomaly_indexes == False)], 'features/test/normal', epoch)

            scores, features, labels, images, reconstruction = evaluate_loop(
                train_loader, record_extreme_cases, record_extreme_cases, loop_name='train evaluation')
            if record_extreme_cases:
                submit_extreme_cases(images, reconstruction, epoch, tag='worst_reconstruction/train')
            writer.add_histogram('log_probs/train', scores, epoch)
            submit_features(features, 'features/train', epoch)
            writer.flush()

        if config.embedding_interval and (epoch + 1) % config.embedding_interval == 0:
            scores, features, labels, images, reconstruction = evaluate_loop(test_loader, record_input_images=True,
                                                                             loop_name='embedding process')
            writer.add_embedding(features, metadata=labels, label_img=images, global_step=epoch,
                                 tag=f'{model_name}/test')
            writer.flush()

        if config.track_weights_interval and epoch % config.track_weights_interval == 0:
            submit_network_weights(epoch)

        train_results = data_feed_loop(train_loader, True, loop_name='train')
        submit_loop_data(train_results, 'train', epoch)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()

        if config.save_interval and (epoch + 1) % config.save_interval == 0:
            model.save(config.models_dir + f'/{model_name}/{model_name}-E{epoch}.pth')

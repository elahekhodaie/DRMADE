import torch as torch
from torch.optim import lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from sklearn.metrics import roc_auc_score
from src.utils.data import DatasetSelection
from src.model.model import DRMADE
import src.config as config

from src.utils.train import Trainer, Action, Loop
import src.utils.train.constants as constants

from .input_transforms import PGDAttackAction, Encode
from .actions import AEForwardPass, EncoderMadeForwardPass
from .loops import RobustAEFeedLoop, RobustMadeFeedLoop

hyper_parameters = {
    'dataset': config.dataset,
    'normal_classes': config.normal_classes,
    'train_batch_size': config.train_batch_size,
    'validation_batch_size': config.validation_batch_size,
    'test_batch_size': config.test_batch_size,

    'made_hidden_layers': config.made_hidden_layers,
    'num_masks': config.made_num_masks,
    'num_mix': config.num_mix,

    'latent_size': config.latent_size,

    'latent_regularization/correlation/factor': 0.,
    'latent_regularization/zero/factor': 0.,
    'latent_regularization/distance/factor': 0.,
    'latent_regularization/variance/factor': 0.,

    'ae_input_pgd/eps': 0.2,
    'ae_input_pgd/iterations': 40,
    'ae_input_pgd/randomize': True,

    'base_lr': config.base_lr,
    'lr_decay': config.lr_decay,
    'lr_half_schedule': config.lr_half_schedule,
}


class DRMADETrainer(Trainer):
    def __init__(
            self,
            drmade=None,
            device=None,
            hparams=dict(),
            name='',
    ):
        # reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.set_default_tensor_type('torch.FloatTensor')

        context = dict()
        context['hparams'] = hparams
        context['max_epoch'] = hparams.get('max_epoch', config.max_epoch)
        # aquiring device cuda if available
        context['device'] = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", context[constants.DEVICE])

        print('loading training data')
        context['train_data'] = DatasetSelection(
            hparams.get('dataset', config.dataset),
            classes=hparams.get('normal_classes', config.normal_classes), train=True)
        print('loading validation data')
        context['validation_data'] = DatasetSelection(
            hparams.get('dataset', config.dataset),
            classes=hparams.get('normal_classes', config.normal_classes), train=False)
        print('loading test data')
        context['test_data'] = DatasetSelection(hparams.get('dataset', config.dataset), train=False)

        context['input_shape'] = context['train_data'].input_shape()

        print('initializing data loaders')
        context['train_loader'] = context['train_data'].get_dataloader(
            shuffle=True, batch_size=hparams.get('train_batch_size', config.train_batch_size))
        context['validation_loader'] = context['validation_data'].get_dataloader(
            shuffle=False, batch_size=hparams.get('validation_batch_size', config.validation_batch_size))
        context['test_loader'] = context['test_data'].get_dataloader(
            shuffle=False, batch_size=hparams.get('test_batch_size', config.test_batch_size))

        print('initializing model')
        context['drmade'] = drmade or DRMADE(
            input_size=context["input_shape"][1],
            num_channels=context["input_shape"][0],
            latent_size=hparams.get('latent_size', config.latent_size),
            made_hidden_layers=hparams.get('made_hidden_layers', config.made_hidden_layers),
            made_natural_ordering=hparams.get('made_natural_ordering', config.made_natural_ordering),
            num_masks=hparams.get('made_num_masks', config.made_num_masks),
            num_mix=hparams.get('num_mix', config.num_mix),
            num_dist_parameters=hparams.get('num_dist_parameters', config.num_dist_parameters),
            distribution=hparams.get('distribution', config.distribution),
            parameters_transform=hparams.get('parameters_transform', config.parameters_transform),
            parameters_min=hparams.get('paramteres_min_value', config.paramteres_min_value),
            encoder_num_layers=hparams.get('encoder_num_layers', config.encoder_num_layers),
            encoder_layers_activation=hparams.get('encoder_layers_activation', config.encoder_layers_activation),
            encoder_latent_activation=hparams.get('encoder_latent_activation', config.encoder_latent_activation),
            encoder_latent_bn=hparams.get('encoder_bn_latent', config.encoder_bn_latent),
            decoder_num_layers=hparams.get('decoder_num_layers', config.decoder_num_layers),
            decoder_layers_activation=hparams.get('decoder_layers_activation', config.decoder_layers_activation),
            decoder_output_activation=hparams.get('decoder_output_activation', config.decoder_output_activation),
        ).to(context[constants.DEVICE])
        context["drmade"].encoder = context["drmade"].encoder.to(context[constants.DEVICE])
        context["drmade"].made = context["drmade"].made.to(context[constants.DEVICE])
        context["drmade"].decoder = context["drmade"].decoder.to(context[constants.DEVICE])

        checkpoint_drmade = hparams.get('checkpoint_drmade', config.checkpoint_drmade)
        if checkpoint_drmade:
            context["drmade"].load(checkpoint_drmade, context[constants.DEVICE])
        checkpoint_encoder = hparams.get('checkpoint_encoder', config.checkpoint_encoder)
        if checkpoint_encoder:
            context["drmade"].encoder.load(checkpoint_encoder, context[constants.DEVICE])
        checkpoint_decoder = hparams.get('checkpoint_decoder', config.checkpoint_drmade)
        if checkpoint_decoder:
            context["drmade"].decoder.load(checkpoint_decoder, context[constants.DEVICE])
        checkpoint_made = hparams.get('checkpoint_made', config.checkpoint_drmade)
        if checkpoint_made:
            context["drmade"].made.load(checkpoint_made, context[constants.DEVICE])

        print(f'model: {context["drmade"].name} was initialized')
        # setting up tensorboard data summerizer
        context['name'] = name or '{}[{}]'.format(
            hparams.get('dataset', config.dataset).__name__,
            ','.join(str(i) for i in hparams.get('normal_classes', config.normal_classes)),
        )
        super(DRMADETrainer, self).__init__(context['name'], context, )

    def setup_writer(self, output_root=None):
        self.context['output_root'] = output_root if output_root else self.context['hparams'].get('output_root',
                                                                                                  config.output_root)
        self.context['models_dir'] = f'{self.context["output_root"]}/models'
        self.context['check_point_saving_dir'] = f'{self.context["models_dir"]}/{self.context["name"]}'

        self.context['runs_dir'] = f'{self.context["output_root"]}/runs'

        # ensuring the existance of output directories
        Path(self.context['output_root']).mkdir(parents=True, exist_ok=True)
        Path(self.context['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.context['check_point_saving_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.context['runs_dir']).mkdir(parents=True, exist_ok=True)

        self.context['writer'] = SummaryWriter(
            log_dir=f'{self.context["runs_dir"]}/{self.context["name"]}')

    def save_model(self, output_path=None):
        output_path = output_path or self.context['check_point_saving_dir']
        self.context['drmade'].save(output_path + f'/{self.context["name"]}-E{self.context["epoch"]}.pth')

    def _evaluate_loop(self, data_loader, record_input_images=False, record_reconstructions=False):
        with torch.no_grad():
            log_prob = torch.Tensor().to(self.context[constants.DEVICE])
            decoder_loss = torch.Tensor().to(self.context[constants.DEVICE])
            reconstructed_images = torch.Tensor().to(self.context[constants.DEVICE])
            features = torch.Tensor().to(self.context[constants.DEVICE])
            labels = np.empty(0, dtype=np.int8)
            input_images = torch.Tensor().to(self.context[constants.DEVICE])
            for batch_idx, (images, label) in enumerate(data_loader):
                images = images.to(self.context[constants.DEVICE])
                if record_input_images:
                    input_images = torch.cat((input_images, images), dim=0)

                output, latent, reconstruction = self.context["drmade"](images)
                decoder_loss = torch.cat(
                    (decoder_loss, self.context["drmade"].decoder.distance(images, reconstruction)), dim=0)
                log_prob = torch.cat((log_prob, self.context["drmade"].made.log_prob_hitmap(latent).sum(1)), dim=0)
                features = torch.cat((features, latent), dim=0)

                if record_reconstructions:
                    reconstructed_images = torch.cat((reconstructed_images, reconstruction), dim=0)
                labels = np.append(labels, label.numpy().astype(np.int8), axis=0)
        return log_prob, decoder_loss, features, labels, input_images, reconstructed_images

    def _submit_latent(self, features, title=''):
        for i in range(features.shape[1]):
            self.context['writer'].add_histogram(f'latent/{title}/{i}', features[:, i], self.context["epoch"])

    def _submit_extreme_reconstructions(self, input_images, reconstructed_images, decoder_loss, title=''):
        num_cases = self.context['hparams'].get('num_extreme_cases', config.num_extreme_cases)
        distance_hitmap = self.context['drmade'].decoder.distance_hitmap(input_images,
                                                                         reconstructed_images).detach().cpu().numpy()

        distance = decoder_loss.detach().cpu().numpy()
        sorted_indexes = np.argsort(distance)
        result_images = np.empty((num_cases * 3, input_images.shape[1], input_images.shape[2], input_images.shape[3]))
        input_images = input_images.cpu()
        reconstructed_images = reconstructed_images.cpu()
        for i, index in enumerate(sorted_indexes[:num_cases]):
            result_images[i * 3] = input_images[index]
            result_images[i * 3 + 1] = reconstructed_images[index]
            result_images[i * 3 + 2] = distance_hitmap[index]
        self.context['writer'].add_images(f'best_reconstruction/{title}', result_images, self.context['epoch'])

        result_images = np.empty((num_cases * 3, input_images.shape[1], input_images.shape[2], input_images.shape[3]))
        for i, index in enumerate(sorted_indexes[-1:-(num_cases + 1):-1]):
            result_images[i * 3] = input_images[index]
            result_images[i * 3 + 1] = reconstructed_images[index]
            result_images[i * 3 + 2] = distance_hitmap[index]
        self.context['writer'].add_images(f'worst_reconstruction/{title}', result_images, self.context['epoch'])

    def _evaluate(self, ):
        record_extreme_cases = self.context['hparams'].get('num_extreme_cases', config.num_extreme_cases)
        log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
            self.context['test_loader'], record_extreme_cases, record_extreme_cases)

        self.context['writer'].add_scalar(
            f'auc/decoder',
            roc_auc_score(y_true=np.isin(labels, config.normal_classes).astype(np.int8),
                          y_score=(-decoder_loss).cpu()), self.context["epoch"])
        self.context['writer'].add_scalar(
            f'auc/made',
            roc_auc_score(y_true=np.isin(labels, config.normal_classes).astype(np.int8),
                          y_score=log_prob.cpu()), self.context["epoch"])
        anomaly_indexes = (
                np.isin(labels, self.context['hparams'].get('normal_classes', config.normal_classes)) == False)
        self.context['writer'].add_histogram(f'loss/decoder/test/anomaly',
                                             decoder_loss[anomaly_indexes], self.context["epoch"])
        self.context['writer'].add_histogram(f'loss/decoder/test/normal',
                                             decoder_loss[(anomaly_indexes == False)],
                                             self.context["epoch"])

        self.context['writer'].add_histogram(f'loss/made/anomaly',
                                             log_prob[anomaly_indexes], self.context["epoch"])
        self.context['writer'].add_histogram(f'loss/made/normal',
                                             log_prob[(anomaly_indexes == False)],
                                             self.context["epoch"])

        if record_extreme_cases:
            self._submit_extreme_reconstructions(
                images[anomaly_indexes], reconstruction[anomaly_indexes], decoder_loss[anomaly_indexes],
                'test/anomaly')
            self._submit_extreme_reconstructions(
                images[(anomaly_indexes == False)], reconstruction[(anomaly_indexes == False)],
                decoder_loss[(anomaly_indexes == False)], 'test/normal')
        self._submit_latent(features[anomaly_indexes], 'test/anomaly')
        self._submit_latent(features[(anomaly_indexes == False)], 'test/normal')

        log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
            self.context['train_loader'], record_extreme_cases, record_extreme_cases)

        self.context['writer'].add_histogram(f'loss/decoder/train',
                                             decoder_loss, self.context["epoch"])
        self.context['writer'].add_histogram(f'loss/made/train',
                                             log_prob, self.context["epoch"])

        self._submit_latent(features, 'train')

        if record_extreme_cases:
            self._submit_extreme_reconstructions(images, reconstruction, decoder_loss, 'train')
        self.context['writer'].flush()

    def _submit_embedding(self):
        log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
            self.context['test_loader'],
            record_input_images=True,
        )
        self.context['writer'].add_embedding(features, metadata=labels, label_img=images,
                                             global_step=self.context['epoch'],
                                             tag=self.context['name'])
        self.context['writer'].flush()

    def train(self):
        evaluation_interval = self.context['hparams'].get('evaluation_interval', config.evaluation_interval)
        embedding_interval = self.context['hparams'].get('embedding_interval', config.embedding_interval)
        save_interval = self.context['hparams'].get('save_interval', config.save_interval)

        for epoch in range(self.context['hparams'].get('start_epoch', 0), self.context['max_epoch']):
            self.context['epoch'] = epoch
            print(f'epoch {self.context["epoch"]:5d}')
            if evaluation_interval and epoch % evaluation_interval == 0:
                self._evaluate()

            if embedding_interval and (epoch + 1) % embedding_interval == 0:
                self._submit_embedding()

            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_model()

            for loop in self.context['loops']:
                if loop.is_active(self.context):
                    self.context[f'{constants.LOOP_PREFIX}{loop.name}/data'] = loop(self.context)
                    loop.submit_loop_data(self.context)

            for scheduler in self.context['schedulers']:
                scheduler.step()

            self.submit_progress()


class RobustAutoEncoderPreTrainer(DRMADETrainer):
    def __init__(self, model=None, device=None, hparams=dict(), name=None):
        super(RobustAutoEncoderPreTrainer, self).__init__(model, device, hparams, name)

        input_limits = self.context['drmade'].decoder.output_limits
        pgd_eps = hparams.get('ae_input_pgd/eps', config.pretrain_ae_pgd_eps)
        pgd_iterations = hparams.get('ae_input_pgd/iterations', config.pretrain_ae_pgd_iterations)
        pgd_alpha = hparams.get('ae_input_pgd/alpha', config.pretrain_ae_pgd_alpha)
        pgd_randomize = hparams.get('ae_input_pgd/randomize', config.pretrain_ae_pgd_randomize)

        latent_limits = self.context['drmade'].encoder.output_limits
        pgd_latent_eps = hparams.get('ae_latent_pgd/eps', config.pretrain_ae_latent_pgd_eps)
        pgd_latent_eps = hparams.get('ae_latent_pgd/iterations', config.pretrain_ae_latent_pgd_iterations)
        pgd_latent_eps = hparams.get('ae_latent_pgd/alpha', config.pretrain_ae_latent_pgd_iterations)
        pgd_latent_randomize = hparams.get('ae_input_pgd/randomize', config.pretrain_ae_latent_pgd_randomize)

        base_lr = hparams.get('base_lr', config.base_lr)
        lr_decay = hparams.get('lr_decay', config.lr_decay)
        lr_half_schedule = hparams.get('lr_half_schedule', config.lr_half_schedule)

        print(f'initializing optimizer Adam - base_lr:{base_lr}')
        optimizer = Adam(
            [
                {'params': self.context['drmade'].encoder.parameters()},
                {'params': self.context['drmade'].decoder.parameters()}
            ], lr=base_lr
        )
        self.context['optimizers'] = [optimizer]
        self.context['optimizer/ae'] = optimizer

        print(f'initializing learning rate scheduler - lr_decay:{lr_decay} half_schedule:{lr_half_schedule}')
        self.context['lr_multiplicative_factor_lambda'] = lambda epoch: 0.5 \
            if (epoch + 1) % lr_half_schedule == 0 else lr_decay
        scheduler = lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=self.context['lr_multiplicative_factor_lambda'], last_epoch=-1)
        self.context['schedulers'] = [scheduler]
        self.context['scheduler/ae'] = scheduler

        self.context[
            'name'] = name or 'PreTrain-{}-{}:{}|pgd-eps{}-iterations{}alpha{}{}|Adam-lr{}-half{}-decay{}'.format(
            self.context['name'],
            self.context['drmade'].encoder.name,
            self.context['drmade'].decoder.name,
            pgd_eps,
            pgd_iterations,
            pgd_alpha,
            'randomized' if pgd_randomize else '',
            base_lr,
            lr_half_schedule,
            lr_decay,
        )
        print("Pre Trainer: ", self.context['name'])
        attacker = PGDAttackAction(
            AEForwardPass('ae'), eps=pgd_eps, iterations=pgd_iterations, alpha=pgd_alpha,
            randomize=pgd_randomize, input_limits=input_limits)

        train_loop = RobustAEFeedLoop(
            name='train',
            data_loader=self.context['train_loader'],
            device=self.context[constants.DEVICE],
            optimizers=('ae',),
            attacker=attacker,
            log_interval=config.log_data_feed_loop_interval,
        )

        validation = RobustAEFeedLoop(
            name='validation',
            data_loader=self.context['validation_loader'],
            device=self.context[constants.DEVICE],
            optimizers=tuple(),
            attacker=attacker,
            interval=hparams.get('validation_interval', config.validation_interval),
            log_interval=config.log_data_feed_loop_interval,
        )

        self.context['loops'] = [validation, train_loop]
        self.setup_writer()


class RobustMadePreTrainer(DRMADETrainer):
    def __init__(self, model=None, device=None, hparams=dict(), name=None):
        super(RobustMadePreTrainer, self).__init__(model, device, hparams, name)

        input_limits = self.context['drmade'].decoder.output_limits
        pgd_eps = hparams.get('encoder_made_input_pgd/eps', config.pretrain_encoder_made_pgd_eps)
        pgd_iterations = hparams.get('encoder_made_input_pgd/iterations', config.pretrain_encoder_made_pgd_iterations)
        pgd_alpha = hparams.get('encoder_made_input_pgd/alpha', config.pretrain_encoder_made_pgd_alpha)
        pgd_randomize = hparams.get('v/randomize', config.pretrain_encoder_made_pgd_randomize)

        base_lr = hparams.get('base_lr', config.base_lr)
        lr_decay = hparams.get('lr_decay', config.lr_decay)
        lr_half_schedule = hparams.get('lr_half_schedule', config.lr_half_schedule)

        print(f'initializing optimizer Adam - base_lr:{base_lr}')
        optimizer = Adam(
            self.context['drmade'].made.parameters(), lr=base_lr
        )
        self.context['optimizers'] = [optimizer]
        self.context['optimizer/made'] = optimizer

        print(f'initializing learning rate scheduler - lr_decay:{lr_decay} half_schedule:{lr_half_schedule}')
        self.context['lr_multiplicative_factor_lambda'] = lambda epoch: 0.5 if \
            (epoch + 1) % lr_half_schedule == 0 else lr_decay
        scheduler = lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=self.context['lr_multiplicative_factor_lambda'], last_epoch=-1)
        self.context['schedulers'] = [scheduler]
        self.context['scheduler/made'] = scheduler

        self.context[
            'name'] = name or 'PreTrain-{}-{}|pgd-eps{}-iterations{}alpha{}{}|Adam-lr{}-half{}-decay{}'.format(
            self.context['name'],
            self.context['drmade'].made.name,
            pgd_eps,
            pgd_iterations,
            pgd_alpha,
            'randomized' if pgd_randomize else '',
            base_lr,
            lr_half_schedule,
            lr_decay,
        )
        print("Pre Trainer: ", self.context['name'])
        attacker = PGDAttackAction(
            EncoderMadeForwardPass('encoder_made'), eps=pgd_eps, iterations=pgd_iterations, alpha=pgd_alpha,
            randomize=pgd_randomize, input_limits=input_limits)

        train_loop = RobustMadeFeedLoop(
            name='train',
            data_loader=self.context['train_loader'],
            device=self.context[constants.DEVICE],
            optimizers=('made',),
            attacker=attacker,
            log_interval=config.log_data_feed_loop_interval,
        )

        validation = RobustMadeFeedLoop(
            name='validation',
            data_loader=self.context['validation_loader'],
            device=self.context[constants.DEVICE],
            optimizers=tuple(),
            attacker=attacker,
            interval=hparams.get('validation_interval', config.validation_interval),
            log_interval=config.log_data_feed_loop_interval,
        )

        self.context['loops'] = [validation, train_loop]
        self.setup_writer()

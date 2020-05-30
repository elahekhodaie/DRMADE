import torch as torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

from sklearn.metrics import roc_auc_score
from src.utils.data import DatasetSelection
from src.models.drmade.model import DRMADE
import src.config as config

from src.utils.train import Trainer
import src.utils.train.constants as constants
import src.models.drmade.config as model_config


class DRMADETrainer(Trainer):
    def __init__(
            self,
            hparams: dict = None,
            name='',
            drmade=None,
            device=None,
    ):
        # reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.set_default_tensor_type('torch.FloatTensor')

        context = dict()
        hparams = hparams or dict()
        context[constants.HPARAMS_DICT] = hparams
        context['normal_classes'] = hparams.get('normal_classes', config.normal_classes)

        # aquiring device cuda if available
        context[constants.DEVICE] = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", context[constants.DEVICE])

        print('loading training data')
        context['train_data'] = DatasetSelection(
            hparams.get('dataset', config.dataset),
            classes=context['normal_classes'], train=True)
        print('loading validation data')
        context['validation_data'] = DatasetSelection(
            hparams.get('dataset', config.dataset),
            classes=context['normal_classes'], train=False)
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

        print('initializing models')
        context['drmade'] = drmade or DRMADE(
            input_size=context["input_shape"][1],
            num_channels=context["input_shape"][0],
            latent_size=hparams.get('latent_size', model_config.latent_size),
            made_hidden_layers=hparams.get('made_hidden_layers', model_config.made_hidden_layers),
            made_natural_ordering=hparams.get('made_natural_ordering', model_config.made_natural_ordering),
            num_masks=hparams.get('made_num_masks', model_config.made_num_masks),
            num_mix=hparams.get('num_mix', model_config.num_mix),
            num_dist_parameters=hparams.get('num_dist_parameters', model_config.num_dist_parameters),
            distribution=hparams.get('distribution', model_config.distribution),
            parameters_transform=hparams.get('parameters_transform', model_config.parameters_transform),
            parameters_min=hparams.get('parameters_min_value', model_config.paramteres_min_value),
            encoder_num_layers=hparams.get('encoder_num_layers', model_config.encoder_num_layers),
            encoder_layers_activation=hparams.get('encoder_layers_activation', model_config.encoder_layers_activation),
            encoder_latent_activation=hparams.get('encoder_latent_activation', model_config.encoder_latent_activation),
            encoder_latent_bn=hparams.get('encoder_bn_latent', model_config.encoder_bn_latent),
            decoder_num_layers=hparams.get('decoder_num_layers', model_config.decoder_num_layers),
            decoder_layers_activation=hparams.get('decoder_layers_activation', model_config.decoder_layers_activation),
            decoder_output_activation=hparams.get('decoder_output_activation', model_config.decoder_output_activation),
            freezed_encoder_layers = hparams.get('freezed_encoder_layers', model_config.freezed_encoder_layers)
        ).to(context[constants.DEVICE])
        context["drmade"].encoder = context["drmade"].encoder.to(context[constants.DEVICE])
        context["drmade"].made = context["drmade"].made.to(context[constants.DEVICE])
        context["drmade"].decoder = context["drmade"].decoder.to(context[constants.DEVICE])

        checkpoint_drmade = hparams.get('checkpoint_drmade', model_config.checkpoint_drmade)
        if checkpoint_drmade:
            context["drmade"].load(checkpoint_drmade, context[constants.DEVICE])
        checkpoint_encoder = hparams.get('checkpoint_encoder', model_config.checkpoint_encoder)
        if checkpoint_encoder:
            context["drmade"].encoder.load(checkpoint_encoder, context[constants.DEVICE])
        checkpoint_decoder = hparams.get('checkpoint_decoder', model_config.checkpoint_drmade)
        if checkpoint_decoder:
            context["drmade"].decoder.load(checkpoint_decoder, context[constants.DEVICE])
        checkpoint_made = hparams.get('checkpoint_made', model_config.checkpoint_drmade)
        if checkpoint_made:
            context["drmade"].made.load(checkpoint_made, context[constants.DEVICE])

        print(f'models: {context["drmade"].name} was initialized')
        # setting up tensorboard data summerizer
        context['name'] = name or '{}[{}]'.format(
            hparams.get('dataset', config.dataset).__name__,
            ','.join(str(i) for i in hparams.get('normal_classes', config.normal_classes)),
        )
        super(DRMADETrainer, self).__init__(context['name'], context, )

    def setup_writer(self, output_root=None):
        self.set('output_root', output_root if output_root else self.context[constants.HPARAMS_DICT].get(
            'output_root', config.output_root))
        self.set('models_dir', f'{self.get("output_root")}/models')
        self.set('check_point_saving_dir', f'{self.context["models_dir"]}/{self.context["name"]}')

        self.set('runs_dir', f'{self.get("output_root")}/runs')

        # ensuring the existance of output directories
        Path(self.get('output_root')).mkdir(parents=True, exist_ok=True)
        Path(self.get('models_dir')).mkdir(parents=True, exist_ok=True)
        Path(self.get('check_point_saving_dir')).mkdir(parents=True, exist_ok=True)
        Path(self.get('runs_dir')).mkdir(parents=True, exist_ok=True)
        self.set('writer', SummaryWriter(log_dir=f'{self.context["runs_dir"]}/{self.context["name"]}'))

    def save_model(self, output_path=None):
        output_path = output_path or self.context['check_point_saving_dir']
        self.get('drmade').save(output_path + f'/{self.get("name")}-E{self.get("epoch")}.pth')

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
        num_cases = self.context[constants.HPARAMS_DICT].get('num_extreme_cases', model_config.num_extreme_cases)
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

    def evaluate(self, ):
        self.get('drmade').eval()
        record_extreme_cases = self.get(constants.HPARAMS_DICT).get('num_extreme_cases', model_config.num_extreme_cases)
        submit_latent_interval = self.get(constants.HPARAMS_DICT).get(
            'submit_latent_interval', model_config.submit_latent_interval)
        evaluate_train_interval = self.get(constants.HPARAMS_DICT).get(
            'evaluate_train_interval', model_config.evaluate_train_interval)
        log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
            self.context['test_loader'], record_extreme_cases, record_extreme_cases)

        self.context['writer'].add_scalar(
            f'auc/decoder',
            roc_auc_score(y_true=np.isin(labels, self.context['normal_classes']).astype(np.int8),
                          y_score=(-decoder_loss).cpu()), self.context["epoch"])
        self.context['writer'].add_scalar(
            f'auc/made',
            roc_auc_score(y_true=np.isin(labels, self.context['normal_classes']).astype(np.int8),
                          y_score=log_prob.cpu()), self.context["epoch"])
        anomaly_indexes = (
                np.isin(labels, self.context[constants.HPARAMS_DICT].get('normal_classes',
                                                                         self.context['normal_classes'])) == False)
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

        if submit_latent_interval and self.get(constants.EPOCH) % submit_latent_interval == 0:
            self._submit_latent(features[anomaly_indexes], 'test/anomaly')
            self._submit_latent(features[(anomaly_indexes == False)], 'test/normal')

        if evaluate_train_interval and self.get(constants.EPOCH) % evaluate_train_interval == 0:
            log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
                self.context['train_loader'], record_extreme_cases, record_extreme_cases)

            self.context['writer'].add_histogram(f'loss/decoder/train',
                                                 decoder_loss, self.context["epoch"])
            self.context['writer'].add_histogram(f'loss/made/train',
                                                 log_prob, self.context["epoch"])
            if submit_latent_interval and self.get(constants.EPOCH) % submit_latent_interval == 0:
                self._submit_latent(features, 'train')

            if record_extreme_cases:
                self._submit_extreme_reconstructions(images, reconstruction, decoder_loss, 'train')
        self.context['writer'].flush()
        self.get('drmade').train()

    def submit_embedding(self):
        log_prob, decoder_loss, features, labels, images, reconstruction = self._evaluate_loop(
            self.context['test_loader'],
            record_input_images=True,
        )
        self.context['writer'].add_embedding(
            features, metadata=labels, label_img=images,
            global_step=self.context['epoch'],
            tag=self.context['name'])
        self.context['writer'].flush()

    def train(self):
        evaluation_interval = self.context[constants.HPARAMS_DICT].get(
            'evaluation_interval', model_config.evaluation_interval)
        embedding_interval = self.context[constants.HPARAMS_DICT].get(
            'embedding_interval', model_config.embedding_interval)
        save_interval = self.context[constants.HPARAMS_DICT].get('save_interval', model_config.save_interval)
        start_epoch = self.context[constants.HPARAMS_DICT].get('start_epoch', 0)
        max_epoch = self.context[constants.HPARAMS_DICT].get('max_epoch', model_config.max_epoch)
        print('Starting Training - intervals:[',
              'evaluation:{}, embedding:{}, save:{}, start_epoch:{}, max_epoch:{} ]'.format(
                  evaluation_interval, embedding_interval, save_interval, start_epoch, max_epoch))

        for epoch in range(start_epoch, max_epoch):
            self.context[constants.EPOCH] = epoch
            print(f'epoch {self.context[constants.EPOCH]:5d}')
            if evaluation_interval and epoch % evaluation_interval == 0:
                if self.verbose:
                    print('\t+ evaluating')
                self.evaluate()

            if embedding_interval and (epoch + 1) % embedding_interval == 0:
                if self.verbose:
                    print('\t+ submitting embedding')
                self.submit_embedding()

            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_model()

            for loop in self.loops_list:
                active = loop.is_active(self.context)
                if self.verbose:
                    print(f'\t+ calling loop {loop.name} - active:{active}')
                if active:
                    self.context[f'{constants.LOOP_PREFIX}{loop.name}/data'] = loop(self.context)
                    if self.verbose:
                        print(f'\t+ submitting loop {loop.name} data')
                    loop.submit_loop_data(self.context)

            for index, scheduler in enumerate(self.schedulers_list):
                if self.verbose:
                    print(f'\t+ scheduler {self.get(f"{constants.SCHEDULER_PREFIX}index/{index}")} step')
                scheduler.step()

            self.submit_progress()

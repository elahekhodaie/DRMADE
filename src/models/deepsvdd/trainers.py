from src.utils.train import Trainer
from src.utils.train import constants
from src.utils.data import DatasetSelection
import torch
import numpy as np
import src.config as config
import src.models.deepsvdd.config as model_config
from .model import DeepSVDD
from torch.optim import lr_scheduler, SGD
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import roc_auc_score
from .actions import Radius
from src.models.drmade.input_transforms import PGDAttackAction

from .loops import RobustDeepSVDDLoop, RobustNCEDeepSVDDLoop
import torchvision.transforms as transforms
import src.models.deepsvdd.custom_transforms as custom_transforms


class DeepSVDDTrainer(Trainer):
    def __init__(
            self,
            model=None,
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
        context['normal_classes'] = hparams.get('normal_classes', config.normal_classes)

        # acquiring device cuda if available
        context[constants.DEVICE] = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", context[constants.DEVICE])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print('loading training data')
        context['train_data'] = DatasetSelection(
            hparams.get('dataset', config.dataset),
            classes=context['normal_classes'], train=True, return_indexes=True,
            transform=transform_train)
        print('loading test data')
        context['test_data'] = DatasetSelection(hparams.get('dataset', config.dataset), train=False,
                                                transform=transform_test)

        context['input_shape'] = context['train_data'].input_shape()

        print('initializing data loaders')
        context['train_loader'] = context['train_data'].get_dataloader(
            shuffle=True, batch_size=hparams.get('train_batch_size', config.train_batch_size))
        context['test_loader'] = context['test_data'].get_dataloader(
            shuffle=False, batch_size=hparams.get('test_batch_size', config.test_batch_size))

        print('initializing models')
        context['models'] = model or DeepSVDD(
            train_data=context['train_data'],
            latent_size=hparams.get('latent_size', model_config.latent_size),
            nce_t=hparams.get('nce_t', model_config.nce_t),
            nce_k=hparams.get('nce_k', model_config.nce_k),
            nce_m=hparams.get('nce_m', model_config.nce_m),
            device=device
        ).to(context[constants.DEVICE])
        context["models"].resnet = context["models"].resnet.to(context[constants.DEVICE])
        print('initializing center - ', end='')
        context["models"].init_center(
            context[constants.DEVICE], init_zero=hparams.get('zero_centered', False))
        print(context["models"].center.mean())
        checkpoint = hparams.get('checkpoint', config.checkpoint_drmade)
        if checkpoint:
            context["models"].load(checkpoint, context[constants.DEVICE])

        print(f'models: {context["models"].name} was initialized')

        base_lr = hparams.get('base_lr', model_config.deepsvdd_sgd_base_lr)
        lr_decay = hparams.get('lr_decay', model_config.deepsvdd_sgd_lr_decay)
        lr_schedule = hparams.get('lr_schedule', model_config.deepsvdd_sgd_schedule)

        sgd_momentum = hparams.get('sgd_momentum', model_config.deepsvdd_sgd_momentum)
        sgd_weight_decay = hparams.get('sgd_weight_deecay', model_config.deepsvdd_sgd_weight_decay)
        pgd_eps = hparams.get('pgd/eps', model_config.deepsvdd_pgd_eps)
        pgd_iterations = hparams.get('pgd/iterations', model_config.deepsvdd_pgd_iterations)
        pgd_alpha = hparams.get('pgd/alpha', model_config.deepsvdd_pgd_alpha)
        pgd_randomize = hparams.get('pgd/randomize', model_config.deepsvdd_pgd_randomize)

        radius_factor = hparams.get('radius_factor', model_config.radius_factor)
        nce_factor = hparams.get('nce_factor', model_config.nce_factor)

        print(f'initializing optimizer SGD - base_lr:{base_lr}')
        optimizer = SGD(
            context['models'].resnet.parameters(), lr=base_lr,
            momentum=sgd_momentum,
            weight_decay=sgd_weight_decay,
        )
        context['optimizers'] = [optimizer]
        context['optimizer/sgd'] = optimizer

        print(f'initializing learning rate scheduler - lr_decay:{lr_decay} half_schedule:{lr_schedule}')
        context['lr_multiplicative_factor_lambda'] = lambda epoch: 0.1 \
            if (epoch + 1) % lr_schedule == 0 else lr_decay
        scheduler = lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=context['lr_multiplicative_factor_lambda'], last_epoch=-1)
        context['schedulers'] = [scheduler]
        context['scheduler/sgd'] = scheduler

        # setting up tensorboard data summerizer
        context['name'] = name or '{}{}-{}{}{}{}|SGDm{}wd{}-baselr{}-decay{}-0.1schedule{}'.format(
            hparams.get('dataset', config.dataset).__name__,
            '{}'.format(
                '' if not context['normal_classes'] else '[' + ','.join(
                    str(i) for i in hparams.get('normal_classes', config.normal_classes)) + ']'
            ),
            context['models'].name,
            f'|NCE{nce_factor}' if nce_factor else '',
            f'|Radius{radius_factor}' if radius_factor else '',
            '' if not pgd_eps else '|pgd-eps{}-iterations{}alpha{}{}'.format(
                pgd_eps, pgd_iterations, pgd_alpha, 'randomized' if pgd_randomize else '',
            ),
            sgd_momentum,
            sgd_weight_decay,
            base_lr,
            lr_decay,
            lr_schedule,
        )
        super(DeepSVDDTrainer, self).__init__(context['name'], context, )

        attacker = PGDAttackAction(
            Radius('radius'), eps=pgd_eps, iterations=pgd_iterations,
            randomize=pgd_randomize, alpha=pgd_alpha)

        train_loop = RobustNCEDeepSVDDLoop(
            name='train',
            data_loader=context['train_loader'],
            device=context[constants.DEVICE],
            optimizers=('sgd',),
            attacker=attacker,
            log_interval=hparams.get('log_interval', config.log_data_feed_loop_interval),
        )

        self.context['loops'] = [train_loop]
        print('setting up writer')
        self.setup_writer()
        print('trainer', context['name'], 'is ready!')

    def setup_writer(self, output_root=None):
        self.context['output_root'] = output_root if output_root else self.context['hparams'].get(
            'output_root', config.output_root)
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
        self.context['models'].save(output_path + f'/{self.context["name"]}-E{self.context["epoch"]}.pth')

    def _evaluate_loop(self, data_loader, record_input_images=False):
        with torch.no_grad():
            features = torch.Tensor().to(self.context[constants.DEVICE])
            labels = np.empty(0, dtype=np.int8)
            input_images = torch.Tensor().to(self.context[constants.DEVICE])
            radii = torch.Tensor().to(self.context[constants.DEVICE])
            for batch_idx, (images, output) in enumerate(data_loader):
                label = output if not data_loader.dataset.return_indexes else output[0]
                images = images.to(self.context[constants.DEVICE])
                if record_input_images:
                    input_images = torch.cat((input_images, images), dim=0)
                radii = torch.cat((radii, self.context['models'].radius_hitmap(images)), dim=0)
                features = torch.cat((features, self.context["models"](images) - self.context['models'].center), dim=0)
                labels = np.append(labels, label.numpy().astype(np.int8), axis=0)
        return radii, features, labels, input_images

    def knn_accuracy(self, K=200, recompute_memory=False):
        self.context['models'].eval()
        total = 0
        testsize = self.context['test_loader'].dataset.__len__()
        trainFeatures = self.context['models'].lemniscate.memory.t()

        trainLabels = torch.LongTensor(self.context['train_loader'].dataset.labels).to(self.context[constants.DEVICE])
        C = trainLabels.max() + 1
        if recompute_memory:
            transform_bak = self.context['train_loader'].dataset.transform
            self.context['train_loader'].dataset.transform = self.context['test_loader'].dataset.transform
            temploader = self.context['train_data'].get_dataloader(shuffle=False, batch_size=100)
            for batch_idx, (images, outputs) in enumerate(temploader):
                images = images.to(self.context[constants.DEVICE])
                targets, indexes = outputs
                batchSize = images.size(0)
                features = self.context['models'](images) - self.context['models'].center
                trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
            trainLabels = torch.LongTensor(temploader.dataset.labels).to(self.context[constants.DEVICE])
            self.context['train_loader'].dataset.transform = transform_bak

        top1 = 0.
        top5 = 0.
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K, C).to(self.context[constants.DEVICE])
            for batch_idx, (images, outputs) in enumerate(self.context['test_loader']):
                images = images.to(self.context[constants.DEVICE])
                outputs = outputs.to(self.context[constants.DEVICE])
                batchSize = images.size(0)
                features = self.context['models'](images) - self.context['models'].center

                dist = torch.mm(features, trainFeatures)

                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1, -1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(self.context['models'].nce_t).exp_()
                probs = torch.sum(
                    torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(outputs.data.view(-1, 1))

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()

                total += outputs.size(0)

        return top1 / total, top5 / total

    def _calculate_accuracy(self):
        top_1_accuracy, top_5_accuracy = 0., 0.
        for images, labels in self.context['test_loader']:
            images = images.to(self.context[constants.DEVICE])
            labels = labels.numpy()
            result = self.context['models'].classify(images)
            for index, top5 in enumerate(result):
                if labels[index] in top5:
                    top_5_accuracy += 1
                if labels[index] == top5[0]:
                    top_1_accuracy += 1
        return top_1_accuracy / (len(self.context['test_loader']) * images.shape[0]), \
               top_5_accuracy / (len(self.context['test_loader']) * images.shape[0])

    def _submit_latent(self, features, title=''):
        for i in range(features.shape[1]):
            self.context['writer'].add_histogram(f'latent/{title}/{i}', features[:, i], self.context["epoch"])

    def evaluate(self, ):
        with torch.no_grad():
            if not self.context['normal_classes']:
                top1, top5 = self.knn_accuracy()
                self.context['writer'].add_scalar(f'accuracy/top1', top1, self.context["epoch"])
                self.context['writer'].add_scalar(f'accuracy/top5', top5, self.context["epoch"])
                print('top1, top5:', top1, top5)

            print('evaluation loop test')
            radii, features, labels, images = self._evaluate_loop(self.context['test_loader'])

            if self.context['normal_classes']:
                self.context['writer'].add_scalar(
                    f'auc/radii',
                    roc_auc_score(y_true=np.isin(labels, self.context['normal_classes']).astype(np.int8),
                                  y_score=(-radii).cpu()), self.context["epoch"])

                anomaly_indexes = (
                        np.isin(labels,
                                self.context['hparams'].get('normal_classes', self.context['normal_classes'])) == False)
                self.context['writer'].add_histogram(f'loss/radii/test/anomaly',
                                                     radii[anomaly_indexes], self.context["epoch"])
                self.context['writer'].add_histogram(f'loss/radii/test/normal',
                                                     radii[(anomaly_indexes == False)],
                                                     self.context["epoch"])

                self._submit_latent(features[anomaly_indexes], 'test/anomaly')
                self._submit_latent(features[(anomaly_indexes == False)], 'test/normal')
            else:
                self._submit_latent(features, 'test')

            print('evaluation loop train')
            radii, features, labels, images = self._evaluate_loop(self.context['train_loader'])

            self.context['writer'].add_histogram(f'loss/radii/train',
                                                 radii, self.context["epoch"])
            self._submit_latent(features, 'train')
            self.context['writer'].flush()

    def submit_embedding(self):
        radii, features, labels, images = self._evaluate_loop(
            self.context['test_loader'], record_input_images=True, )
        self.context['writer'].add_embedding(features, metadata=labels, label_img=images,
                                             global_step=self.context['epoch'],
                                             tag=self.context['name'])
        self.context['writer'].flush()

    def train(self):
        evaluation_interval = self.get(constants.HPARAMS_DICT).get(
            'evaluation_interval', model_config.evaluation_interval)
        embedding_interval = self.get(constants.HPARAMS_DICT).get('embedding_interval', model_config.embedding_interval)
        save_interval = self.get(constants.HPARAMS_DICT).get('save_interval', model_config.save_interval)

        for epoch in range(self.context['hparams'].get('start_epoch', 0), self.context['max_epoch']):
            self.context['epoch'] = epoch
            print(f'epoch {self.context["epoch"]:5d}')
            if evaluation_interval and epoch % evaluation_interval == 0:
                self.evaluate()

            if embedding_interval and (epoch + 1) % embedding_interval == 0:
                self.submit_embedding()

            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_model()

            for loop in self.context['loops']:
                if loop.is_active(self.context):
                    self.context[f'{constants.LOOP_PREFIX}{loop.name}/data'] = loop(self.context)
                    loop.submit_loop_data(self.context)

            for scheduler in self.context['schedulers']:
                scheduler.step()

            self.submit_progress()

from torch.optim import lr_scheduler, Adam

import src.config as config

import src.utils.train.constants as constants
import src.models.drmade.config as config

from src.models.drmade.loops import RobustMadeFeedLoop
from src.models.drmade.trainers.base_trainer import DRMADETrainer


class partialConnectedEncoder(DRMADETrainer):

    # for the partial connected encoder mode , if we don't need the pgd attack mode
    # we can simply change the attach epsilon in config to zero

    def __init__(self, hparams = None ,  name = None,  model = None, device= None):
        super(partialConnectedEncoder, self).__init__(hparams, name, model, device, )
        hparams = self.get(constants.HPARAMS_DICT)
        name = 'partialConnectedEncoder'

        # base_lr = hparams.get('base_lr', config.base_lr)
        # lr_decay = hparams.get('lr_decay', config.lr_decay)
        # lr_schedule = hparams.get('lr_schedule', config.lr_schedule)

        # pgd encoder decoder inputs
        input_limits = self.get('drmade').decoder.output_limits
        pgd_eps = hparams.get('pgd/eps', model_config.pretrain_ae_pgd_eps)
        pgd_iterations = hparams.get('pgd/iterations', model_config.pretrain_ae_pgd_iterations)
        pgd_alpha = hparams.get('pgd/alpha', model_config.pretrain_ae_pgd_alpha)
        pgd_randomize = hparams.get('pgd/randomize', model_config.pretrain_ae_pgd_randomize)
        pgd_input = {'eps': pgd_eps, 'iterations': pgd_iterations, 'alpha': pgd_alpha, 'randomize': pgd_randomize,
                     'input_limits': input_limits}
        # pgd decoder inputs
        latent_input_limits = self.get('drmade').encoder.output_limits
        pgd_latent_eps = hparams.get('pgd_latent/eps', model_config.pretrain_ae_latent_pgd_eps)
        pgd_latent_iterations = hparams.get('pgd_latent/iterations', model_config.pretrain_ae_latent_pgd_iterations)
        pgd_latent_alpha = hparams.get('pgd_latent/alpha', model_config.pretrain_ae_latent_pgd_alpha)
        pgd_latent_randomize = hparams.get('pgd_latent/randomize', model_config.pretrain_ae_latent_pgd_randomize)
        pgd_latent = {'eps': pgd_latent_eps, 'iterations': pgd_latent_iterations, 'alpha': pgd_latent_alpha,
                      'randomize': pgd_latent_randomize, 'input_limits': latent_input_limits}
        base_lr = hparams.get('base_lr', model_config.base_lr)
        lr_decay = hparams.get('lr_decay', model_config.lr_decay)
        lr_schedule = hparams.get('lr_schedule', model_config.lr_schedule)

        print ("freezing some layers in partial_trainer :")


        for param in self.get('drmade').encoder.convs[0].parameters():
            param.requires_grad = False

        for param in self.get('drmade').encoder.convs[2].parameters():
            param.requires_grad = False

        for param in self.get('drmade').encoder.convs[4].parameters():
            param.requires_grad = False

        print(f'initializing optimizer Adam - base_lr:{base_lr}')
        optimizer = Adam(
            [
                {'params': self.get('drmade').encoder.parameters()},
                {'params': self.get('drmade').made.parameters()}
            ], lr=base_lr
        )
        self.add_optimizer( 'partial_train', optimizer)

        print(f'initializing learning rate scheduler - lr_decay:{lr_decay} schedule:{lr_schedule}')
        self.set('lr_multiplicative_factor_lambda', lambda epoch: 0.5 if (epoch + 1) % lr_schedule == 0 else lr_decay)
        self.add_scheduler('partial_train', lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=self.get('lr_multiplicative_factor_lambda'), last_epoch=-1))

        self.set(constants.TRAINER_NAME, name or 'PreTrain-{}-{}:{}|{}{}Adam-lr{}-schedule{}-decay{}'.format(
            self.get(constants.TRAINER_NAME), self.get('drmade').encoder.name, self.get('drmade').decoder.name,
            '' if not pgd_eps else 'pgd-eps{}-iterations{}alpha{}{}|'.format(
                pgd_eps, pgd_iterations, pgd_alpha, 'randomized' if pgd_randomize else '', ),
            '' if not pgd_latent_eps else 'pgd-latent-eps{}-iterations{}alpha{}{}|'.format(
                pgd_latent_eps, pgd_latent_iterations, pgd_latent_alpha,
                'randomized' if pgd_latent_randomize else '', ), base_lr, lr_schedule, lr_decay), replace=True)
        print("Pre Trainer: ", self.get(constants.TRAINER_NAME))

        self.add_loop(RobustMadeFeedLoop(
            name='train',
            data_loader=self.context['train_loader'],
            device=self.context[constants.DEVICE],
            optimizers=('partial_train',),
            pgd_input=pgd_input,
            pgd_latent=pgd_latent,
            log_interval=hparams.get('log_interval', config.log_data_feed_loop_interval)))

        self.add_loop(RobustMadeFeedLoop(
            name='validation',
            data_loader=self.context['validation_loader'],
            device=self.context[constants.DEVICE],
            optimizers=tuple(),
            pgd_input=pgd_input,
            pgd_latent=pgd_latent,
            interval=hparams.get('validation_interval', model_config.validation_interval),
            log_interval=hparams.get('log_interval', config.log_data_feed_loop_interval)))

        self.setup_writer()

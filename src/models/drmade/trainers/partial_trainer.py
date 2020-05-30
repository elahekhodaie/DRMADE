from torch.optim import lr_scheduler, Adam

import src.config as config

import src.utils.train.constants as constants
import src.models.drmade.config as model_config
from src.models.drmade.loops import RobustMadeFeedLoop

from src.models.drmade.trainers import RobustMadePreTrainer


class partialConnectedEncoder(RobustMadePreTrainer):
    def __init__(self, hparams: dict = None, name=None, model=None, device=None, ):
        freezed_layers = hparams.get('freezed_encoder_layers', [])

        base_lr = hparams.get('base_lr', model_config.base_lr)
        lr_decay = hparams.get('lr_decay', model_config.lr_decay)
        lr_schedule = hparams.get('lr_schedule', model_config.lr_schedule)

        super().__init__(self, hparams, name, model, device)
      #  super(RobustMadePreTrainer, self).__init__(hparams, name, model, device)

        hparams = self.get(constants.HPARAMS_DICT)
        hparams['submit_latent_interval'] = 1
        hparams['submit_latent_interval'] = 1

        hparams['embedding_interval'] = 512

        name = name or 'Freezed_layer_train-{}-{}-{}|Adam-lr{}-schedule{}-decay{}'.format(
            self.get(constants.TRAINER_NAME), self.get('drmade').made.name,
            self.get('drmade').encoder.name,
            base_lr, lr_schedule, lr_decay, )
        print("partial_connected", self.get(constants.TRAINER_NAME))

        print(f're-initializing optimizer Adam - base_lr:{base_lr}')
        optimizer = Adam(
            [{'params': self.get('drmade').encoder.parameters()},
             {'params': self.get('drmade').made.parameters()}], lr=base_lr)

        self.add_optimizer('partial_connected', optimizer, replace=True)

        # freezed folent
        print('unfreezing encoder')
        for parameter in self.get('drmade').encoder.parameters():
            parameter.requires_grad = True

        assert max(freezed_layers)<len(self.get('drmade').encoder.conv_layers)
        for i in freezed_layers:
            print(f'frozen layers are {i}')
            for parameter in self.get('drmade').encoder.conv_layers[i].parameters():
                parameter.requires_grad = False

        self.setup_writer()
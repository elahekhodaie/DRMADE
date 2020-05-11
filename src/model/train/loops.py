from src.utils.train import Loop
from src.utils.train import constants
import src.config as config
import torch
from .actions import EncoderMadeForwardPass, AEForwardPass
from .input_transforms import PGDAttackAction, Encode


class RobustAEFeedLoop(Loop):
    def __init__(self, name, data_loader, device, optimizers=None, attacker=None, interval=1, log_interval=0):
        attacker = attacker or PGDAttackAction(
            AEForwardPass('ae'), eps=config.pretrain_ae_pgd_eps, iterations=config.pretrain_ae_pgd_iterations,
            randomize=config.pretrain_ae_pgd_randomize, alpha=config.pretrain_ae_pgd_alpha)
        super(RobustAEFeedLoop, self).__init__(
            name,
            data_loader,
            device,
            input_transforms=(attacker,),
            loss_actions=(AEForwardPass('ae', 'pgd-ae'),),
            optimizers=optimizers,
            log_interval=log_interval
        )
        self.interval = interval

    def is_active(self, context: dict = None, **kwargs):
        return self.interval and context['epoch'] % self.interval == 0


class RobustMadeFeedLoop(Loop):
    def __init__(self, name, data_loader, device, optimizers=None, attacker=None, interval=1, log_interval=0):
        attacker = attacker or PGDAttackAction(
            EncoderMadeForwardPass('encoder_made'), eps=config.pretrain_made_pgd_eps,
            iterations=config.pretrain_made_pgd_iterations, randomize=config.pretrain_made_pgd_randomize,
            alpha=config.pretrain_made_pgd_alpha, transformed_input='encoded_inputs'
        )
        super(RobustMadeFeedLoop, self).__init__(
            name,
            data_loader,
            device,
            input_transforms=(Encode(), attacker,),
            loss_actions=(EncoderMadeForwardPass('negative_log_prob', 'pgd-encoder_made'),),
            optimizers=optimizers,
            log_interval=log_interval
        )
        self.interval = interval

    def is_active(self, context: dict = None, **kwargs):
        return self.interval and context['epoch'] % self.interval == 0

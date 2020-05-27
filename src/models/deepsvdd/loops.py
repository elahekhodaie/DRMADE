from src.utils.train import Loop
from src.utils.train import constants
import src.config as config
import src.models.deepsvdd.config as model_config
import torch
from .actions import LogProb, UpdateMemory, Radius, IterationDifference, NCE
from src.models.drmade.input_transforms import PGDAttackAction


class RobustDeepSVDDLoop(Loop):
    def __init__(self, name, data_loader, device, optimizers=None, attacker=None, interval=1, log_interval=0):
        attacker = attacker or PGDAttackAction(
            Radius('radius'), eps=model_config.deepsvdd_pgd_eps, iterations=model_config.deepsvdd_pgd_iterations,
            randomize=model_config.deepsvdd_pgd_randomize, alpha=model_config.deepsvdd_pgd_alpha)
        super(RobustDeepSVDDLoop, self).__init__(
            name,
            data_loader,
            device,
            input_transforms=(attacker,),
            loss_actions=(
                LogProb('log-prob'), IterationDifference('iteration_difference'), Radius('radius', 'pgd-radius'),
            ),
            after_optimization_context_actions=(UpdateMemory('update_memory'),),
            optimizers=optimizers,
            log_interval=log_interval
        )
        self.interval = interval

    def is_active(self, context: dict = None, **kwargs):
        return self.interval and context['epoch'] % self.interval == 0


class RobustNCEDeepSVDDLoop(Loop):
    def __init__(self, name, data_loader, device, optimizers=None, attacker=None, interval=1, log_interval=0):
        attacker = attacker or PGDAttackAction(
            Radius('radius'), eps=model_config.deepsvdd_pgd_eps, iterations=model_config.deepsvdd_pgd_iterations,
            randomize=model_config.deepsvdd_pgd_randomize, alpha=model_config.deepsvdd_pgd_alpha)
        super(RobustNCEDeepSVDDLoop, self).__init__(
            name,
            data_loader,
            device,
            input_transforms=(attacker,),
            loss_actions=(
                NCE('nce'), Radius('radius', 'pgd-radius'),
            ),
            optimizers=optimizers,
            log_interval=log_interval
        )
        self.interval = interval

    def is_active(self, context: dict = None, **kwargs):
        return self.interval and context['epoch'] % self.interval == 0

from src.utils.train import Loop
from src.utils.train import constants
import src.config as config
import torch
from .actions import LogProb, UpdateMemory, Radius, IterationDifference
from src.model.drmade.input_transforms import PGDAttackAction


class RobustDeepSVDDLoop(Loop):
    def __init__(self, name, data_loader, device, optimizers=None, attacker=None, interval=1, log_interval=0):
        attacker = attacker or PGDAttackAction(
            Radius('radius'), eps=config.deepsvdd_pgd_eps, iterations=config.deepsvdd_pgd_iterations,
            randomize=config.deepsvdd_pgd_randomize, alpha=config.deepsvdd_pgd_alpha)
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

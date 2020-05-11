import torch
from .constants import *


class Trainer:
    def __init__(
            self,
            name: str = '',
            context: dict = None,
            loops=tuple()
    ):

        self.context = context or dict()
        if EPOCH not in self.context:
            self.context[EPOCH] = 0
        if 'name' not in self.context:
            self.context['name'] = name

        self.context['loops'] = loops
        for loop in loops:
            self.context[f'{LOOP_PREFIX}{loop.name}'] = loop
            self.context[f'{LOOP_PREFIX}{loop.name}/data'] = None

    def submit_progress(self):
        for item in self.context:
            if item.startswith(OPTIMIZER_PREFIX):
                self.context["writer"].add_scalar(
                    f'{LEARNING_RATE_PREFIX}{item[len(f"{OPTIMIZER_PREFIX}"):]}',
                    self.context[item].param_groups[0]["lr"], self.context[EPOCH]
                )

    def stop_training(self):
        raise NotImplemented

    def train(self):
        while not self.stop_training():
            print(f'epoch {self.context[EPOCH]:5d}')
            for loop in self.context['loops']:
                if loop.is_active(self.context):
                    self.context[f'{LOOP_PREFIX}{loop.name}/data'] = loop(self.context)
                    loop.submit_loop_data(self.context)
            self.submit_progress()
            self.context[EPOCH] += 1

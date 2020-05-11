from .constants import *
import torch
import time


class Loop:
    def __init__(
            self,
            name: str = '',
            data_loader=None,
            device=None,
            loss_actions=tuple(),  # a list of LossActions
            context_actions=tuple(),  # a list of ContextActions
            input_transforms=tuple(),
            optimizers=tuple(),  # a list of optimizer names
            log_interval=0,
            no_grad=False,
    ):
        self.name = name
        self.device = device
        self.optimizers = optimizers
        self.data_loader = data_loader
        self.input_transforms = input_transforms
        self.context_actions = context_actions
        self.loss_actions = loss_actions

        self.log_interval = log_interval
        self.no_grad = no_grad
        self.loop_data = dict()

    def no_grad_active(self, context=None, **kwargs):
        if self.no_grad is None or self.no_grad is False:
            return False
        if callable(self.no_grad):
            return self.no_grad(context, **kwargs)
        return self.no_grad

    def submit_loop_data(self, context):
        for item in self.loop_data:
            if item.startswith(f'{RESULT_PREFIX}{SCALAR_PREFIX}'):
                context["writer"].add_scalar(
                    f'{item[len(f"{RESULT_PREFIX}{SCALAR_PREFIX}"):]}/{self.name}',
                    self.loop_data[item], context[EPOCH]
                )
            if item.startswith(f'{ACTION_FACTOR_PREFIX}'):
                context["writer"].add_scalar(
                    f'{item}/{self.name}',
                    self.loop_data[item], context[EPOCH]
                )

    def is_active(self, context: dict = None, **kwargs):
        return True

    def __call__(
            self,
            context=None,
            data_loader=None,
            model=None,
            device=None,
            **kwargs
    ):
        loop_data = dict()
        assert (data_loader is None and self.data_loader is not None) or data_loader is not None, \
            f'no data_loader specified - loop:{self.name}'
        data_loader = self.data_loader or data_loader
        assert (device is None and self.device is not None) or device is not None or (
                DEVICE in context and context[DEVICE]), f'no device specified - loop:{self.name}'
        device = self.device or device
        device = device or context[DEVICE]

        time_ = time.time()

        if self.loss_actions:
            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}loss'] = 0.

        for action in self.loss_actions:
            loop_data[f'{ACTION_FACTOR_PREFIX}{action.name}'] = 0.
            loop_data[f'{ACTION_PREFIX}{action.name}/calls_count'] = 0
            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}{action.name}'] = 0.

        with torch.set_grad_enabled(not self.no_grad_active(context, **kwargs)):
            for batch_idx, (inputs, outputs) in enumerate(data_loader):
                inputs = inputs.to(device)
                loop_data['inputs'] = inputs
                loop_data['outputs'] = outputs

                for input_tranfrom in self.input_transforms:
                    results = input_tranfrom(
                        inputs, outputs, context, loop_data=loop_data, **kwargs)
                    if isinstance(results, dict):
                        for name, value in results.items():
                            loop_data[f'{TRANSFORM_PREFIX}{input_tranfrom.name}{name}'] = value
                    else:
                        if input_tranfrom.changes_input_directly(context, loop_data, **kwargs):
                            loop_data[f'{TRANSFORM_PREFIX}{input_tranfrom.name}'] = results

                loss = 0.
                for action in self.loss_actions:
                    factor = action.factor(context, loop_data=loop_data, **kwargs)
                    if factor:
                        loop_data[f'{ACTION_PREFIX}{action.name}/calls_count'] += 1
                        loop_data[f'{ACTION_FACTOR_PREFIX}{action.name}'] += factor
                        result = action(
                            inputs, outputs, context, loop_data=loop_data, **kwargs)
                        loss += factor * result
                        loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}{action.name}'] += result

                loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}loss'] += loss

                for action in self.context_actions:
                    if action.is_active(context, loop_data=loop_data, **kwargs):
                        action(inputs, outputs, context, loop_data=loop_data, **kwargs)

                if self.optimizers and self.loss_actions:
                    for optimizer_name in self.optimizers:
                        context[f'{OPTIMIZER_PREFIX}{optimizer_name}'].zero_grad()
                    loss.backward()
                    for optimizer_name in self.optimizers:
                        context[f'{OPTIMIZER_PREFIX}{optimizer_name}'].step()

                if self.log_interval and (batch_idx + 1) % self.log_interval == 0:
                    print(
                        '\t{}\t{:3d}/{:3d} - loss : {:.4f}, time : {:.3f}s'.format(
                            self.name,
                            batch_idx + 1,
                            len(data_loader),
                            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}loss'] / (1 + batch_idx),
                            time.time() - time_)
                    )
                    time_ = time.time()

        for item in loop_data:
            if item.startswith(f'{RESULT_PREFIX}{SCALAR_PREFIX}'):
                loop_data[item] /= len(data_loader)
            if item.startswith(f'{ACTION_FACTOR_PREFIX}'):
                loop_data[item] /= loop_data[f'{ACTION_PREFIX}{item[len(ACTION_FACTOR_PREFIX):]}/calls_count']
        self.loop_data = loop_data
        return loop_data

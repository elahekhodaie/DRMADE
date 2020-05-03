import time
import torch as torch
from collections.abc import Iterable

TRANSFORM_PREFIX = 'transform/'
OPTIMIZER_PREFIX = 'optimizer/'
SCHEDULER_PREFIX = 'scheduler/'
MODEL_PREFIX = 'model/'
ACTION_PREFIX = 'action/'
LOOP_PREFIX = 'loop/'

RESULT_PREFIX = 'result/'
SCALAR_PREFIX = 'scalar/'
ACTION_FACTOR_PREFIX = 'action_factor/'
LEARNING_RATE_PREFIX = 'learning_rate/'
DEVICE = 'device'
EPOCH = 'epoch'
HPARAM_PREFIX = 'hparam/'


class InputTransform:
    def __init__(
            self,
            name,
            transformed_input=None,  # can be callable ( context, loop_data, **kwargs)
    ):
        self.name = name
        self.transformed_input = transformed_input

    def is_active(self, context=None, loop_data: dict = None, **kwargs):
        return True

    def dependency_input_names(self, context=None, loop_data: dict = None, **kwargs):
        result = []
        if not self.transformed_input:
            return tuple()
        if callable(self.transformed_input):
            result = self.transformed_input(context, loop_data, **kwargs)
        elif isinstance(self.transformed_input, Iterable):
            result = list(self.transformed_input)
        if not isinstance(result, Iterable) and not result:
            return []
        if not isinstance(result, list):
            return [result]
        return result

    def dependency_inputs(self, context: dict = None, loop_data: dict = None, **kwargs):
        assert loop_data is not None, f'loop_data not provided - InputTransform: {self.name}'
        return {
            dep: loop_data.get(f'{TRANSFORM_PREFIX}{dep}', None) for dep in self.dependency_input_names(
                context, loop_data, **kwargs)
        }

    def transform(self, inputs, outputs=None, context=None, loop_data: dict = None,
                  dependency_inputs: dict = None, **kwargs):
        raise NotImplemented

    def __call__(self, inputs, outputs=None, context=None, loop_data: dict = None,
                 dependency_inputs: dict = None, **kwargs):
        dependency_inputs = dependency_inputs or self.dependency_inputs(context, loop_data)
        if self.is_active(context, loop_data, **kwargs):
            return self.transform(inputs, outputs, context, loop_data, dependency_inputs, **kwargs)
        else:
            return inputs


class Action:  # returns a single factor of loss / new_context (   input or ...
    def __init__(
            self,
            name,
            transformed_input=None,  # can be callable ( context, loop_data, **kwargs)
    ):
        self.name = name
        self.transformed_input = transformed_input

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        if context is not None:
            return context[f'{ACTION_FACTOR_PREFIX}{self.name}'] if self.is_active(context, **kwargs) else 0.
        return 1. if self.is_active(context, **kwargs) else 0.

    def dependency_input_names(self, context=None, loop_data: dict = None, **kwargs):
        result = []
        if not self.transformed_input:
            return []
        if callable(self.transformed_input):
            result = self.transformed_input(context, loop_data, **kwargs)
        elif isinstance(self.transformed_input, Iterable):
            result = list(self.transformed_input)
        if not isinstance(result, Iterable) and not result:
            return []
        if not isinstance(result, list):
            return [result]
        return result

    def dependency_inputs(self, context: dict = None, loop_data: dict = None, **kwargs):
        assert loop_data is not None, f'loop_data not provided - Action: {self.name}'
        return {
            dep: loop_data.get(f'{TRANSFORM_PREFIX}{dep}', None) for dep in self.dependency_input_names(
                context, loop_data, **kwargs)
        }

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None,
               dependency_inputs: dict = None, **kwargs):
        raise NotImplemented

    def __call__(self, inputs, outputs=None, context=None, loop_data: dict = None,
                 dependency_inputs: dict = None, **kwargs):
        dependency_inputs = dependency_inputs or self.dependency_inputs(context, loop_data)
        result = self.action(inputs, outputs, context, loop_data, dependency_inputs, **kwargs)
        if result:
            return result


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
            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}{action.name}'] = 0.

        with torch.set_grad_enabled(not self.no_grad_active(context, **kwargs)):
            for batch_idx, (inputs, outputs) in enumerate(data_loader):
                inputs = inputs.to(device)
                for input_tranfrom in self.input_transforms:
                    results = input_tranfrom(
                        inputs, outputs, context, loop_data=loop_data, **kwargs)
                    if isinstance(results, dict):
                        for name, value in results.items():
                            loop_data[f'{TRANSFORM_PREFIX}{input_tranfrom.name}{name}'] = value
                    else:
                        loop_data[f'{TRANSFORM_PREFIX}{input_tranfrom.name}'] = results

                loss = 0.
                for action in self.loss_actions:
                    factor = action.factor(context, loop_data=loop_data, **kwargs)
                    loop_data[f'{ACTION_FACTOR_PREFIX}{action.name}'] += factor / len(data_loader)
                    if factor:
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
                            self.name, batch_idx, len(data_loader),
                            loop_data[f'{RESULT_PREFIX}{SCALAR_PREFIX}loss'] / (1 + batch_idx),
                            time.time() - time_)
                    )
                    time_ = time.time()

        for item in loop_data:
            if item.startswith(f'{RESULT_PREFIX}{SCALAR_PREFIX}') == 0:
                loop_data[item] = loop_data[item] / len(data_loader)
        self.loop_data = loop_data
        return loop_data


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

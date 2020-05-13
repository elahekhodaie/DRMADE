from src.utils.train import InputTransform
from src.utils.train import constants
import src.config as config
import torch


class PGDAttackAction(InputTransform):
    def __init__(self, action, eps=0.2, alpha=0.05, iterations=20, randomize=True,
                 input_limits=config.input_limits, transformed_input=None, ):
        super(PGDAttackAction, self).__init__(f'pgd-{action.name}')
        self.action = action
        self.eps = eps
        self.alpha = alpha
        self.iterations = iterations
        self.randomize = randomize
        self.input_limits = input_limits
        self.transformed_input = transformed_input

    def dependency_inputs(self, context=None, loop_data: dict = None, **kwargs):
        return tuple()

    def is_active(self, context=None, loop_data: dict = None, **kwargs):
        return self.eps != 0.

    def transform(self, inputs, outputs=None, context=None, loop_data: dict = None, *args, **kwargs):
        assert self.transformed_input is None or f'{constants.TRANSFORM_PREFIX}{self.transformed_input}' in loop_data, \
            f'InputTransform/{self.name} transformed input {self.transformed_input} not specified in loop_data'
        if self.transformed_input:
            inputs = loop_data.get[f'{constants.TRANSFORM_PREFIX}{self.transformed_input}'].detach()
            inputs.requires_grad = False
        if self.randomize:
            delta = torch.rand_like(inputs, requires_grad=True)
            delta.data = delta.data * 2 * self.eps - self.eps
        else:
            delta = torch.zeros_like(inputs, requires_grad=True)
        for i in range(self.iterations):
            loss = self.action(inputs + delta, outputs, context=context, loop_data=loop_data)
            loss.backward()
            delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(-self.eps, self.eps)
            delta.grad.zero_()
        if self.input_limits:
            return (inputs + delta.detach()).clamp(*self.input_limits)
        return inputs + delta.detach()


class Encode(InputTransform):
    def __init__(self, transformed_input=None):
        super(Encode, self).__init__(f'encoded_inputs')
        self.transformed_input = transformed_input

    def dependency_inputs(self, context=None, loop_data: dict = None, **kwargs):
        return tuple()

    def is_active(self, context=None, loop_data: dict = None, **kwargs):
        return True

    def transform(self, inputs, outputs=None, context=None, loop_data: dict = None, *args, **kwargs):
        assert self.transformed_input is None or f'{constants.TRANSFORM_PREFIX}{self.transformed_input}' in loop_data, \
            f'InputTransform/{self.name} transformed input {self.transformed_input} not specified in loop_data'
        if self.transformed_input:
            inputs = loop_data[f'{constants.TRANSFORM_PREFIX}{self.transformed_input}']
        return context['drmade'].encoder(inputs)


class UniformNoiseInput(InputTransform):
    def __init__(self, factor=0.2, input_limits=config.input_limits):
        super(UniformNoiseInput, self).__init__('uniform_noise_input')
        self.factor = factor
        self.input_limits = input_limits

    def transform(self, inputs, outputs=None, context=None, loop_data: dict = None,
                  dependency_inputs: dict = None, **kwargs):
        result = inputs + self.factor * (
                2 * torch.DoubleTensor(*inputs.shape).to(context[constants.DEVICE]).uniform_() - 1)
        if self.input_limits:
            return result.clamp(*self.input_limits)
        return result

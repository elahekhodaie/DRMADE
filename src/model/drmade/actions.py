from src.utils.train import Action
from src.utils.train import constants
import src.config as config
import torch


class LatentRegularization(Action):
    def __init__(self, name, function):
        super(LatentRegularization, self).__init__(f'latent_regularization/{name}')
        self.function = function

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        return context['hparams'].get(f'{self.name}/factor', 0.)

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return context['hparams'].get(f'{self.name}/factor', 0.)

    def __call__(self, inputs, outputs=None, context=None, loop_data: dict = None, **kwargs):
        return self.function(context)(loop_data.get('false_features', loop_data.features))


latent_cor_action = LatentRegularization('correlation',
                                         lambda context: context["drmade"].encoder.latent_cor_regularization)
latent_distance_action = LatentRegularization('distance',
                                              lambda context: context["drmade"].encoder.latent_distance_regularization)
latent_zero_action = LatentRegularization('zero',
                                          lambda context: context["drmade"].encoder.latent_zero_regularization)
latent_var_action = LatentRegularization('variance',
                                         lambda context: context["drmade"].encoder.latent_var_regularization)


class AEForwardPass(Action):
    def __init__(self, name='', transformed_input=None):
        super(AEForwardPass, self).__init__(name, transformed_input)

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        return 1.

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        if dependency_inputs:
            inputs = dependency_inputs.get(f'pgd-{self.name}', inputs)
        features = None
        if dependency_inputs:
            features = dependency_inputs.get(f'pgd-latent', context["drmade"].encoder(inputs))
        if features is None:
            features = context["drmade"].encoder(inputs)
        reconstructions = context["drmade"].decoder(features)
        return context["drmade"].decoder.distance(loop_data['inputs'], reconstructions).sum()


class MadeForwardPass(Action):
    def __init__(self, name='', transformed_input=None):
        super(MadeForwardPass, self).__init__(name, transformed_input)

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        return 1.

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        assert self.transformed_input is None or f'{constants.TRANSFORM_PREFIX}{self.transformed_input}' in loop_data, \
            f'Action/{self.name} transformed input {self.transformed_input} not specified in loop_data'
        if self.transformed_input:
            inputs = loop_data[f'{constants.TRANSFORM_PREFIX}{self.transformed_input}']
        return -context["drmade"].made.log_prob(inputs)


class EncoderMadeForwardPass(Action):
    def __init__(self, name='', transformed_input=None):
        super(EncoderMadeForwardPass, self).__init__(name, transformed_input)

    def factor(self, context: dict = None, loop_data: dict = None, **kwargs):
        return 1.

    def is_active(self, context: dict = None, loop_data: dict = None, **kwargs):
        return True

    def action(self, inputs, outputs=None, context=None, loop_data: dict = None, dependency_inputs=None, **kwargs):
        assert self.transformed_input is None or f'{constants.TRANSFORM_PREFIX}{self.transformed_input}' in loop_data, \
            f'Action/{self.name} transformed input {self.transformed_input} not specified in loop_data'
        if self.transformed_input:
            inputs = loop_data[f'{constants.TRANSFORM_PREFIX}{self.transformed_input}']

        return -context["drmade"].made.log_prob(context['drmade'].encoder(inputs))

import torch
import time
from .constants import *
from collections.abc import Iterable


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
            return context.get(f'{ACTION_FACTOR_PREFIX}{self.name}', 0) if self.is_active(context, **kwargs) else 0.
        return 1. if self.is_active(context, **kwargs) else 0.

    def dependency_input_names(self, context=None, loop_data: dict = None, **kwargs):
        result = []
        if not self.transformed_input:
            return []
        if callable(self.transformed_input):
            result = self.transformed_input(context, loop_data, **kwargs)
        elif isinstance(self.transformed_input, Iterable):
            if isinstance(self.transformed_input, str):
                result = [self.transformed_input]
            else:
                result = list(self.transformed_input)
        if not isinstance(result, Iterable) and not result:
            return []
        if not isinstance(result, list) and result:
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

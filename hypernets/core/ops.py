# -*- coding:utf-8 -*-
"""

"""

from .search_space import *


class HyperInput(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _build(self):
        pass

    def _compile(self, inputs):
        return inputs


class Identity(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _build(self):
        pass

    def _compile(self, inputs):
        return inputs


class ConnectionSpace(ModuleSpace):
    def __init__(self, dynamic_fn, keep_link=False, space=None, name=None, **hyperparams):
        self.dynamic_fn = dynamic_fn
        self.keep_link = keep_link
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _on_params_ready(self):
        with self.space.as_default():
            input, output = self.dynamic_fn(self)
            if not all([input, output]):
                # node removed
                if self.keep_link:
                    # Only one input and one output are allowed
                    inputs = self.space.get_inputs(self)
                    outputs = self.space.get_outputs(self)
                    if len(inputs) > 1 or len(outputs) > 1:
                        raise ValueError("Only one input and one output are allowed.")
                    self.space.disconnect_all(self)
                    outputs[0](inputs[0])
                else:
                    self.space.disconnect_all(self)

            elif all([input, output]):
                self.space.reroute_to(self, input)
                self.space.reroute_from(self, output)
            else:
                raise ValueError('input or output is None.')


class Optional(ConnectionSpace):

    def __init__(self, module, keep_link=False, space=None, name=None, hp_opt=None):
        assert isinstance(module, ModuleSpace), f'{module} is not a valid Module. '
        self._module = module
        self.hp_opt = hp_opt if hp_opt is not None else Bool()
        ConnectionSpace.__init__(self, self.optional_fn, keep_link, space, name, hp_opt=self.hp_opt)

    def optional_fn(self, m):
        if self.hp_opt.value:
            return self._module, self._module
        else:
            return None, None


class Or(ConnectionSpace):
    def __init__(self, module_list, keep_link=False, space=None, name=None):
        assert isinstance(module_list, list), f'module_list must be a List.'
        assert len(module_list) > 0, f'module_list contains at least 1 Module.'
        assert all([isinstance(m, ModuleSpace) for m in module_list]), 'module_list can only contain Module.'
        self.hp_or = Choice(list(range(len(module_list))))
        self._module_list = module_list
        ConnectionSpace.__init__(self, self.or_fn, keep_link, space, name, hp_or=self.hp_or)

    def or_fn(self, m):
        module = self._module_list[self.hp_or.value]
        return module, module


class Sequential(ConnectionSpace):
    def __init__(self, module_list, keep_link=False, space=None, name=None):
        assert isinstance(module_list, list), f'module_list must be a List.'
        assert len(module_list) > 0, f'module_list contains at least 1 Module.'
        assert all([isinstance(m, ModuleSpace) for m in module_list]), 'module_list can only contain Module.'
        self._module_list = module_list
        self.hp_lazy = Choice([0])
        ConnectionSpace.__init__(self, self.sequential_fn, keep_link, space, name, hp_lazy=self.hp_lazy)

    def sequential_fn(self, m):
        last = self._module_list[0]
        for i in range(1, len(self._module_list)):
            self._module_list[i](last)
            last = self._module_list[i]
        input = self._module_list[0]
        output = last
        return input, output


class InputChoice(ConnectionSpace):
    def __init__(self, connection_num, max_chosen_num=0, keep_link=False, space=None, name=None):
        self.hp_choice = MultipleChoice(list(range(connection_num)), max_chosen_num)
        self.connection_num = connection_num
        self.max_chosen_num = max_chosen_num
        ConnectionSpace.__init__(self, None, keep_link, space, name, hp_choice=self.hp_choice)

    def _on_params_ready(self):
        with self.space.as_default():
            inputs = self.space.get_inputs(self)
            for i, input in enumerate(inputs):
                if i not in self.hp_choice.value:
                    self.space.disconnect(input, self)

            outputs = self.space.get_outputs(self)
            for output in outputs:
                self.space.reroute_to(self, output)

            self.space.disconnect_all(self)


class Reduction(ModuleSpace):
    def __init__(self, reduction_op, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.reduction_op = reduction_op

    def _build(self):
        pv = self.param_values
        if pv.get('name') is None:
            pv['name'] = self.name
        self.compile_fn = self.reduction_op(**pv)
        self.is_built = True

    def _compile(self, inputs):
        return self.compile_fn(inputs)

    def _on_params_ready(self):
        self._build()

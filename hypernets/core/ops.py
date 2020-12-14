# -*- coding:utf-8 -*-
"""

"""
from .search_space import ModuleSpace, Bool, Choice, MultipleChoice
import itertools


class HyperInput(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        pass

    def _forward(self, inputs):
        return inputs


class Identity(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        pass

    def _forward(self, inputs):
        return inputs


class ConnectionSpace(ModuleSpace):
    def __init__(self, dynamic_fn, keep_link=False, space=None, name=None, **hyperparams):
        self.dynamic_fn = dynamic_fn
        self.keep_link = keep_link
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _on_params_ready(self):
        with self.space.as_default():
            try:
                self.space.scope.entry(self.id)
                input, output = self.dynamic_fn(self)
                if not all([input, output]) or (
                        isinstance(input, list) and isinstance(output, list) and (len(input) <= 0 or len(output) <= 0)):
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
            finally:
                self.space.scope.exit()

    def connect_module_or_subgraph(self, from_module, to_module, support_subgraph=True):
        if not support_subgraph:
            assert isinstance(from_module, ModuleSpace)
            assert isinstance(to_module, ModuleSpace)
            assert self.space.is_isolated_module(from_module), f'`from_module` is not an isolated module. '
            assert self.space.is_isolated_module(from_module), f'`to_module` is not an isolated module. '
            return to_module(from_module)
        else:
            assert isinstance(from_module, (ModuleSpace, list))
            assert isinstance(to_module, (ModuleSpace, list))

            if isinstance(from_module, ModuleSpace):
                real_from = [from_module]
            else:
                assert len(from_module) > 0, f'`from_module` contains at least 1 element.'
                # If from_module is a list, take any module to get the outputs of the subgraph
                real_from = from_module[:1]

            if not self.space.is_isolated_module(real_from[0]):
                real_from = list(self.space.get_sub_graph_outputs(real_from[0]))

            if isinstance(to_module, ModuleSpace):
                real_to = [to_module]
            else:
                assert len(to_module) > 0, f'`from_module` contains at least 1 element.'
                # If to_module is a list, take any module to get the inputs of the subgraph
                real_to = to_module[:1]

            if not self.space.is_isolated_module(real_to[0]):
                real_to = list(self.space.get_sub_graph_inputs(real_to[0]))
            for m_to in real_to:
                for m_from in real_from:
                    m_to(m_from)
            return to_module


class Optional(ConnectionSpace):

    def __init__(self, module, keep_link=False, space=None, name=None, hp_opt=None):
        assert isinstance(module, ModuleSpace), f'{module} is not a valid Module. '
        self._module = module
        self.hp_opt = hp_opt if hp_opt is not None else Bool()
        ConnectionSpace.__init__(self, self.optional_fn, keep_link, space, name, hp_opt=self.hp_opt)

    def optional_fn(self, m):
        if self.hp_opt.value:
            input = self.space.get_sub_graph_inputs(self._module)
            output = self.space.get_sub_graph_outputs(self._module)
            return input, output
        else:
            return None, None


class ModuleChoice(ConnectionSpace):
    def __init__(self, module_list, keep_link=False, space=None, name=None, hp_or=None):
        assert isinstance(module_list, list), f'module_list must be a List.'
        assert len(module_list) > 0, f'module_list contains at least 1 Module.'
        assert all([isinstance(m, ModuleSpace) for m in module_list]), 'module_list can only contain Module.'

        self.hp_or = hp_or if hp_or is not None else Choice(list(range(len(module_list))))
        self._module_list = module_list
        ConnectionSpace.__init__(self, self.or_fn, keep_link, space, name, hp_or=self.hp_or)

    def or_fn(self, m):
        module = self._module_list[self.hp_or.value]
        input = self.space.get_sub_graph_inputs(module)
        output = self.space.get_sub_graph_outputs(module)
        return input, output


class Sequential(ConnectionSpace):
    def __init__(self, module_list, keep_link=False, space=None, name=None):
        assert isinstance(module_list, list), f'module_list must be a List.'
        assert len(module_list) > 0, f'module_list contains at least 1 Module.'
        assert all([isinstance(m, (ModuleSpace, list)) for m in
                    module_list]), 'module_list can only contains ModuleSpace or list.'
        self._module_list = module_list
        self.hp_lazy = Choice([0])
        ConnectionSpace.__init__(self, self.sequential_fn, keep_link, space, name, hp_lazy=self.hp_lazy)

    def sequential_fn(self, m):
        last = self._module_list[0]
        for i in range(1, len(self._module_list)):
            self.connect_module_or_subgraph(last, self._module_list[i])
            # self._module_list[i](last)
            last = self._module_list[i]
        input = self.space.get_sub_graph_inputs(last)
        output = self.space.get_sub_graph_outputs(last)
        return input, output


class Permutation(ConnectionSpace):
    def __init__(self, module_list, keep_link=False, space=None, name=None, hp_seq=None):
        assert isinstance(module_list, list), f'module_list must be a List.'
        assert len(module_list) > 1, f'module_list contains at least 2 Module.'
        assert all([isinstance(m, ModuleSpace) for m in module_list]), 'module_list can only contain Module.'
        self._module_list = module_list

        if hp_seq is None:
            p = itertools.permutations(range(len(module_list)))
            all_seq = []
            for seq in p:
                all_seq.append(seq)
            self.hp_all_seq = Choice(all_seq)
        else:
            self.hp_all_seq = hp_seq

        ConnectionSpace.__init__(self, self.permutation_fn, keep_link, space, name, hp_all_seq=self.hp_all_seq)

    def permutation_fn(self, m):
        seq = self.hp_all_seq.value
        # input = None
        last = None
        for i in seq:
            # if input is None:
            #    input = self._module_list[i]
            if last is not None:
                self.connect_module_or_subgraph(last, self._module_list[i])
                # self._module_list[i](last)
            last = self._module_list[i]
        input = self.space.get_sub_graph_inputs(last)
        output = self.space.get_sub_graph_outputs(last)
        return input, output


class Repeat(ConnectionSpace):
    def __init__(self, module_fn, keep_link=False, space=None, name=None, repeat_times=[1]):
        assert callable(module_fn), f'{module_fn} is not a callable object. '
        assert isinstance(repeat_times, list), f'`repeat_num_choices` must be a list.'
        assert all([isinstance(c, int) for c in
                    repeat_times]), f'All of the element in `repeat_num_choices` must be integer.'
        assert all(
            [c > 0 for c in repeat_times]), f'All of the element in `repeat_num_choices` must be greater than 0.'
        self.module_fn = module_fn
        self.hp_repeat_times = Choice(repeat_times)
        ConnectionSpace.__init__(self, self.repeat_fn, keep_link, space, name, hp_repeat_times=self.hp_repeat_times)

    def repeat_fn(self, m):
        repeat_times = self.hp_repeat_times.value
        module_list = [self.module_fn(step) for step in range(repeat_times)]
        last = module_list[0]
        for i in range(1, len(module_list)):
            self.connect_module_or_subgraph(last, module_list[i])
            # module_list[i](last)
            last = module_list[i]
        input = self.space.get_sub_graph_inputs(last)
        output = self.space.get_sub_graph_outputs(last)
        return input, output


class InputChoice(ConnectionSpace):
    def __init__(self, inputs, num_chosen_most=0, num_chosen_least=1, keep_link=False, space=None, name=None,
                 hp_choice=None):
        assert isinstance(inputs, list)
        connection_num = len(inputs)
        assert connection_num >= num_chosen_least, f'`inputs` contains at least {num_chosen_least} item.'
        self.inputs = inputs
        self.hp_choice = hp_choice if hp_choice is not None else MultipleChoice(list(range(connection_num)),
                                                                                num_chosen_most, num_chosen_least)
        ConnectionSpace.__init__(self, None, keep_link, space, name, hp_choice=self.hp_choice)

    def _on_params_ready(self):
        with self.space.as_default():
            for input in self.inputs:
                self.space.disconnect(input, self)
            for i, input in enumerate(self.inputs):
                if i in self.hp_choice.value:
                    self.space.connect(input, self)

            outputs = self.space.get_outputs(self)
            for output in outputs:
                self.space.reroute_to(self, output)

            self.space.disconnect_all(self)


class ConnectLooseEnd(ConnectionSpace):
    def __init__(self, inputs, keep_link=False, space=None, name=None):
        assert isinstance(inputs, list)
        self.inputs = inputs
        self.hp_lazy = Choice([0])
        ConnectionSpace.__init__(self, None, keep_link, space, name, hp_lazy=self.hp_lazy)

    def _on_params_ready(self):
        with self.space.as_default():
            for input in self.inputs:
                outputs = self.space.get_outputs(input)
                # It's not a loose end if the input has other downstream node excepts this node
                if len(set(outputs) - {self}) > 0:
                    self.space.disconnect(input, self)

            outputs = self.space.get_outputs(self)
            for output in outputs:
                self.space.reroute_to(self, output)

            self.space.disconnect_all(self)


class Reduction(ModuleSpace):
    def __init__(self, reduction_op, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.reduction_op = reduction_op

    def _compile(self):
        pv = self.param_values
        if pv.get('name') is None:
            pv['name'] = self.name
        self.compile_fn = self.reduction_op(**pv)

    def _forward(self, inputs):
        return self.compile_fn(inputs)

    def _on_params_ready(self):
        self._compile()

# -*- coding:utf-8 -*-
"""

"""

import numpy as np
import threading
import contextlib
import queue
import copy
from collections import OrderedDict
from .mutables import Mutable, MutableScope
from ..utils.common import generate_id


class HyperNode(Mutable):
    def __init__(self, space=None, name=None):
        self._space = space if space is not None else get_default_space()
        Mutable.__init__(self, self._space.scope, name)
        self._space.add_node(self)

    @property
    def space(self):
        return self._space


class HyperSpace(Mutable):

    def __init__(self, scope=None, name=None):
        if scope is None:
            scope = MutableScope()
        Mutable.__init__(self, scope, name)
        self.edges = set()
        self.modules = set()
        self.hyper_params = set()
        self._is_compiled = False
        self.space_id = generate_id()

    @property
    def type(self):
        return 'DAG'

    def as_default(self):
        return _default_space_stack.get_controller(self)

    @property
    def all_assigned(self):
        all_assigned = self.traverse(lambda m: m.all_assigned, direction='backward')
        return all_assigned

    @property
    def unassigned_iterator(self):
        visited = {}
        while not self.all_assigned:
            for p in self.get_assignable_params():
                if not p.assigned:
                    if visited.get(p):
                        visited[p] += 1
                        if visited[p] > 10:
                            return
                            # raise RuntimeError('Too many attempts to get assignable params')
                    else:
                        visited[p] = 1
                        yield p

    def add_node(self, node):
        if isinstance(node, Module):
            self.modules.add(node)
        elif isinstance(node, ParameterSpace):
            self.hyper_params.add(node)
        else:
            raise ValueError(f"Not supported node:{node}")
        self.__dict__[node.id] = node

    def compile_space(self):
        space_copy = copy.deepcopy(self)
        space_copy._compile_space()
        return space_copy

    def _compile_space(self):
        assert not self._is_compiled, 'HyperSpace does not allow to compile repeatedly.'
        space_out = []

        def compile_module(module):
            inputs = self.get_inputs(module)
            if len(inputs) <= 0:
                module.compile()
            elif len(inputs) == 1:
                module.compile(inputs[0].output)
            else:
                module.compile([m.output for m in inputs])

            outputs = self.get_outputs(module)
            if len(outputs) <= 0:
                space_out.append(module.output)
            return True

        self.traverse(compile_module, direction='forward')
        self._is_compiled = True

    def traverse(self, fn, direction='forward'):
        if direction == 'forward':
            fn_inputs = self.get_inputs
            fn_outputs = self.get_outputs
        elif direction == 'backward':
            fn_inputs = self.get_outputs
            fn_outputs = self.get_inputs
        else:
            raise ValueError(f'Not supported direction:{direction}')

        standby = queue.Queue()
        visited = set()
        finished = set()
        for m in fn_inputs():
            standby.put(m)
            visited.add(m)

        while not standby.empty():
            m_todo = standby.get()
            inputs = fn_inputs(m_todo)
            ready = True
            for mi in inputs:
                if mi not in finished:
                    ready = False
                    break

            if not ready:
                visited.remove(m_todo)
                continue

            is_continues = fn(m_todo)

            if not is_continues:
                return False
            finished.add(m_todo)
            for m in fn_outputs(m_todo):
                if not m in visited:
                    standby.put(m)
                    visited.add(m)
        return True

    def connect(self, from_module, to_module):
        self.edges.add((from_module, to_module))

    def disconnect(self, from_module, to_module):
        found = False
        for f, t in self.edges:
            if f == from_module and t == to_module:
                found = True
                break
        self.edges.remove((from_module, to_module))

    def disconnect_all(self, module):
        found = set()
        for f, t in self.edges:
            if f == module or t == module:
                found.add((f, t))
        for f, t in found:
            self.edges.remove((f, t))

    def reroute_to(self, old_module, new_module):
        found = set()
        for f, t in self.edges:
            if t == old_module:
                found.add((f, t))

        for f, t in found:
            self.edges.remove((f, t))
            self.edges.add((f, new_module))

    def reroute_from(self, old_module, new_module):
        found = set()
        for f, t in self.edges:
            if f == old_module:
                found.add((f, t))

        for f, t in found:
            self.edges.remove((f, t))
            self.edges.add((new_module, t))

    def replace_route(self, old_module, new_module):
        self.reroute_to(old_module, new_module)
        self.reroute_from(old_module, new_module)

    def get_inputs(self, module=None):
        inputs = set()
        if module is None:
            for m in self.modules:
                no_in = True
                has_out = False
                for from_module, to_module in self.edges:
                    if m == from_module:
                        has_out = True
                    if m == to_module:
                        no_in = False
                        break
                # discard orphan node
                if no_in and has_out:
                    inputs.add(m)
            if len(inputs) == 0:
                if len(self.modules) == 1:
                    inputs = self.modules.copy()
                else:
                    raise ValueError('Graph is not connected.')

        else:
            for from_module, to_module in self.edges:
                if module == to_module:
                    inputs.add(from_module)

        return sorted(inputs, key=lambda m: m.id)

    def get_outputs(self, module=None):
        outputs = set()
        if module is None:
            for m in self.modules:
                no_out = True
                has_input = False
                for from_module, to_module in self.edges:
                    if m == to_module:
                        has_input = True
                    if m == from_module:
                        no_out = False
                        break
                # discard orphan node
                if no_out and has_input:
                    outputs.add(m)
            if len(outputs) == 0:
                if len(self.modules) == 1:
                    outputs = self.modules.copy()
                else:
                    raise ValueError('Graph is not connected.')
        else:
            for from_module, to_module in self.edges:
                if module == from_module:
                    outputs.add(to_module)
        return sorted(outputs, key=lambda m: m.id)

    def get_io(self, module=None):
        inputs = self.get_inputs(module)
        outputs = self.get_outputs(module)
        return inputs, outputs

    def random_sample(self):
        for hp in self.unassigned_iterator:
            hp.random_sample()

    def get_assignable_params(self, traverse_direction='forward'):
        assignables = []

        def append_params(m):
            ps = m.get_assignable_params()
            for p in ps:
                if p not in assignables:
                    assignables.append(p)
            return True

        self.traverse(append_params, direction=traverse_direction)
        return assignables

    def get_assignable_param_values(self, traverse_direction='forward'):
        assignables = self.get_assignable_params()
        return {p.id: p.value for p in assignables}

    def get_all_params(self, traverse_direction='forward'):
        all = []

        def append_params(m):
            ps = m.get_all_params()
            for p in ps:
                if p not in all:
                    all.append(p)
            return True

        self.traverse(append_params, direction=traverse_direction)
        return all

    def params_summary(self, only_assignable=True, traverse_direction='forward', line_width=60):
        print(f'\n{(line_width + 2) * "-"}')
        if only_assignable:
            params = self.get_assignable_params(traverse_direction)
        else:
            params = self.get_all_params(traverse_direction)

        for i, hp in enumerate(params):
            print(
                f'({i}) {hp.alias}:{(line_width - len(str(i) + "() " + hp.alias + str(hp.value))) * " "}{hp.value}')
        print(f'{(line_width + 2) * "-"}')


class DefaultStack(threading.local):
    def __init__(self):
        super(DefaultStack, self).__init__()
        self.enforce_nesting = True
        self.stack = []

    def get_default(self):
        return self.stack[-1] if len(self.stack) >= 1 else None

    def reset(self):
        self.stack = []

    def is_cleared(self):
        return not self.stack

    @contextlib.contextmanager
    def get_controller(self, default):
        self.stack.append(default)
        try:
            yield default
        finally:
            if self.stack:
                if self.enforce_nesting:
                    if self.stack[-1] is not default:
                        raise AssertionError(
                            "Nesting violated for default stack of %s objects" %
                            type(default))
                    self.stack.pop()
                else:
                    self.stack.remove(default)


class DefaultSpaceStack(DefaultStack):
    def __init__(self):
        super(DefaultSpaceStack, self).__init__()
        self._global_default_space = None

    def get_default(self):
        """Override that returns a global default if the stack is empty."""
        default = super(DefaultSpaceStack, self).get_default()
        if default is None:
            default = self._global_default()
        return default

    def _global_default(self):
        if self._global_default_space is None:
            self._global_default_space = HyperSpace()
        return self._global_default_space

    def reset(self):
        super(DefaultSpaceStack, self).reset()
        self._global_default_space = None


_default_space_stack = DefaultSpaceStack()


def get_default_space():
    return _default_space_stack.get_default()


class ParameterSpace(HyperNode):
    def __init__(self, space=None, name=None):
        HyperNode.__init__(self, space, name)
        self._assigned = False
        self._value = None
        self.references = set()

    @property
    def is_mutable(self):
        return True

    @property
    def type(self):
        return 'Param'

    @property
    def assigned(self):
        return self._assigned

    @property
    def value(self):
        return self._value

    def random_sample(self, assign=True):
        value = self._random_sample()
        if assign:
            self.assign(value)
        return value

    def _random_sample(self):
        raise NotImplementedError

    def assign(self, value):
        assert not self._assigned
        self._check(value)
        self._assigned = True
        self._value = value

        for m in self.references:
            m.update()

    def attach(self, mutable, alias=None):
        self.references.add(mutable)
        if alias is not None:
            all = [] if self.alias is None else [self.alias]
            all.append(mutable.name + '.' + alias)
            self.alias = ','.join(all)

    def detach(self, mutable):
        self.references.remove(mutable)

    def _check(self, value):
        pass


class Int(ParameterSpace):
    def __init__(self, low, high, random_state=np.random.RandomState(), space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        assert isinstance(low, int) and isinstance(high, int), '`low` and `high` must be a int.'
        assert low < high, '`low` must less than `high`.'
        self.low = low
        self.high = high
        self.random_state = random_state

    def _random_sample(self):
        return self.random_state.randint(self.low, self.high)

    def _check(self, value):
        assert value >= self.low and value <= self.high


class Real(ParameterSpace):
    def __init__(self, low, high, q=None, prior="uniform", random_state=np.random.RandomState(), space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        low = float(low)
        high = float(high)
        assert low < high, '`low` must less than `high`.'
        self.low = low
        self.high = high
        self.q = q
        self.prior = prior
        self.random_state = random_state

    def _random_sample(self):
        if self.prior == "uniform":
            assert self.high >= self.low, 'Upper bound must be larger than lower bound'
            value = self.random_state.uniform(self.low, self.high)
            self._check(value)
        elif self.prior == "log_uniform":
            assert self.low > 0, 'Lower bound must be positive'
            value = np.exp(self.random_state.uniform(self.low, self.high))
        elif self.prior == "q_uniform":
            assert self.q is not None, 'q cannot be None'
            value = np.clip(
                np.round(self.random_state.uniform(self.low, self.high) / self.q) * self.q, self.low, self.high)
            self._check(value)
        else:
            raise ValueError(f'Not supported prior:{self.prior}')
        return value

    def _check(self, value):
        if self.prior == "log_uniform":
            assert value >= np.exp(self.low) and value <= np.exp(self.high)
        else:
            assert value >= self.low and value <= self.high


class Choice(ParameterSpace):
    def __init__(self, options, random_state=np.random.RandomState(), space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        assert isinstance(options, list), '`options` must be a List.'
        assert len(options) > 0, '`options` contains at least one item.'
        self.options = options
        self.random_state = random_state

    @property
    def is_mutable(self):
        return len(self.options) > 1

    def _random_sample(self):
        return self.random_state.choice(self.options)

    def _check(self, value):
        assert value in self.options


class MultipleChoice(ParameterSpace):
    def __init__(self, options, max_chosen_num=0, random_state=np.random.RandomState(), space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        assert isinstance(options, list), '`options` must be a List.'
        assert len(options) > 1, '`options` contains at least 2 item.'
        self.options = options
        self.max_chosen_num = max_chosen_num
        self.random_state = random_state

    def _random_sample(self):
        options = self.options.copy()
        options_state = [[True, False] for _ in range(len(options))]
        loop_counter = 0
        while True:
            loop_counter += 1
            values = []
            self.random_state.shuffle(options)
            for i in range(len(options)):
                if self.random_state.choice(options_state[i]):
                    values.append(options[i])
                    if self.max_chosen_num > 0 and len(values) >= self.max_chosen_num:
                        break
            if len(values) > 0:
                break
            if loop_counter > 100:
                raise TimeoutError(f'Retry counter exceeded: {loop_counter}')
        return values

    def _check(self, value):
        assert isinstance(value, list)
        assert len(value) > 0, 'value contains at least 1 item.'
        assert (self.max_chosen_num == 0 or self.max_chosen_num <= len(self.options))
        assert all([v in self.options for v in value])


class Bool(Choice):
    def __init__(self, random_state=np.random.RandomState(), space=None, name=None):
        Choice.__init__(self, [True, False], random_state, space, name)


class Constant(ParameterSpace):
    def __init__(self, value, space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        self.assign(value)

    @property
    def is_mutable(self):
        return False


class Dynamic(ParameterSpace):
    def __init__(self, lambda_fn, space=None, name=None, **param_dict):
        ParameterSpace.__init__(self, space, name)
        self._lambda_fn = lambda_fn
        self._param_dict = {}

        for n, p in param_dict.items():
            if isinstance(p, Dynamic):
                raise ValueError('Dynamic cannot be nested.')
            self._param_dict[n] = p
            p.attach(self, n)

        self.update()

    @property
    def is_mutable(self):
        return False

    def update(self):
        if all(p.assigned for p in self._param_dict.values()):
            args = {name: p.value for name, p in self._param_dict.items()}
            value = self._lambda_fn(args)
            self.assign(value)

    @property
    def param_dict(self):
        return self._param_dict


class Cascade(ParameterSpace):
    def __init__(self, lambda_fn, space=None, name=None, **param_dict):
        ParameterSpace.__init__(self, space, name)
        self._lambda_fn = lambda_fn
        self._param_dict = {}

        for n, p in param_dict.items():
            self._param_dict[n] = p
            p.attach(self, n)

        self.update()

    @property
    def is_mutable(self):
        return False

    @property
    def assigned(self):
        if self.value is not None:
            if isinstance(self.value, ParameterSpace):
                return self.value.assigned
            else:
                return True

    def update(self):
        if all(p.assigned for p in self._param_dict.values()):
            args = {name: p.value for name, p in self._param_dict.items()}
            name, value = self._lambda_fn(args, self.space)
            assert isinstance(value, ParameterSpace), 'The value of `Cascade` must be a ParameterSpace.'
            for m in self.references:
                if isinstance(m, Module):
                    m.add_parameters(**{name: value})
                elif isinstance(m, ParameterSpace):
                    value.attach(m)
            self.assign(value)

    @property
    def param_dict(self):
        return self._param_dict


class Module(HyperNode):
    def __init__(self, space=None, name=None, **hyperparams):
        HyperNode.__init__(self, space, name)
        self._hyper_params = OrderedDict()
        self.is_built = False
        self._is_params_ready = False
        self._is_compiled = False
        self._output = None
        self.space.add_node(self)
        self.add_parameters(**hyperparams)
        self.update()

    def __call__(self, *args, **kwargs):
        assert len(args) > 0
        m = args[0]
        if isinstance(m, Module):
            self.space.connect(m, self)
        elif isinstance(m, list):
            for mi in m:
                self.space.connect(mi, self)
        return self

    def connect(self, module_or_list):
        if isinstance(module_or_list, Module):
            self.space.connect(self, module_or_list)
        elif isinstance(module_or_list, list):
            assert len(module_or_list) > 0, f'module_or_list contains at least 1 Module.'
            assert all([isinstance(m, Module) for m in module_or_list]), 'module_or_list can only contain Module.'
            for m in module_or_list:
                self.space.connect(self, m)
        else:
            raise ValueError(f'module_or_list is neither Module nor List.')
        return self

    @property
    def type(self):
        return 'Module'

    @property
    def param_values(self):
        return {name: p.value for name, p in self._hyper_params.items()}

    @property
    def hyper_params(self):
        return self._hyper_params

    def get_assignable_params(self):
        assignables = []
        for name, p in self._hyper_params.items():
            if isinstance(p, Constant):
                continue
            elif isinstance(p, (Dynamic, Cascade)):
                for dp in p.param_dict.values():
                    if not isinstance(dp, (Dynamic, Cascade)) and not dp in assignables:
                        assignables.append(dp)
            else:
                if not p in assignables:
                    assignables.append(p)
        return assignables

    def get_all_params(self):
        all = []
        for name, p in self._hyper_params.items():
            if not p in all:
                all.append(p)
            if isinstance(p, (Dynamic, Cascade)):
                for dp in p.param_dict.values():
                    if not dp in all:
                        all.append(dp)
        return all

    @property
    def output(self):
        return self._output

    @property
    def is_compiled(self):
        return self._is_compiled

    @property
    def is_params_ready(self):
        return self._is_params_ready

    @property
    def all_assigned(self):
        for hp in self._hyper_params.values():
            if not hp.assigned:
                return False
        return True

    def compile(self, inputs=None):
        if not self.is_built:
            self._build()
            self.is_built = True
        self._output = self._compile(inputs)
        self._is_compiled = True
        return self._output

    def _compile(self, inputs):
        raise NotImplementedError

    def _build(self):
        raise NotImplementedError

    def _on_params_ready(self):
        pass

    def add_parameters(self, **hyperparameters):
        for name, param in hyperparameters.items():
            if not isinstance(param, ParameterSpace):
                param = Constant(param)
            if self._hyper_params.get(name) is not None:
                raise ValueError(f'Parameter `{name}` has existed.')
            self._hyper_params[name] = param
            param.attach(self, name)

    def update(self):
        if all(p.assigned for p in self._hyper_params.values()):
            self._is_params_ready = True
            self._on_params_ready()

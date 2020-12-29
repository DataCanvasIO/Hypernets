# -*- coding:utf-8 -*-
"""

"""

import numpy as np
import hashlib
import threading
import contextlib
import queue
import copy
import time
from collections import OrderedDict
from .mutables import Mutable, MutableScope
from ..utils.common import generate_id, combinations
from ..utils import logging

logger = logging.get_logger(__name__)


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
        self._inputs = set()
        self._outputs = set()
        self._assigned_params_stack = []
        self._is_compiled = False
        self.space_id = generate_id()

    @property
    def type(self):
        return 'DAG'

    def as_default(self):
        return _default_space_stack.get_controller(self)

    @property
    def all_assigned(self):
        all_assigned = self.traverse(lambda m: m.all_assigned, direction='forward', discard_isolated_node=False)
        return all_assigned

    @property
    def assigned_params_stack(self):
        return self._assigned_params_stack

    def push_assigned_param(self, param):
        self._assigned_params_stack.append(param)

    @property
    def params_iterator(self):
        visited = {}
        while not self.all_assigned:
            for p in self.get_unassigned_params():
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
        if isinstance(node, ModuleSpace):
            self.modules.add(node)
        elif isinstance(node, ParameterSpace):
            self.hyper_params.add(node)
        else:
            raise ValueError(f"Not supported node:{node}")
        self.__dict__[node.id] = node

    def compile(self, deepcopy=True):
        if deepcopy:
            space = copy.deepcopy(self)
        else:
            space = self
        space._compile_space()
        return space

    def _compile_space(self):
        assert not self._is_compiled, 'HyperSpace does not allow to compile repeatedly.'
        space_out = []
        counter = []
        start_ts = time.time()

        def compile_module(module):
            module.compile()
            counter.append(0)
            if len(self.get_outputs(module)) <= 0:
                space_out.append(module)
            return True

        self.traverse(compile_module, direction='forward')
        end_ts = time.time()
        if logger.is_info_enabled():
            logger.debug(f'Compile Space: compiled {len(counter)} modules in {end_ts - start_ts} seconds.')
        self._is_compiled = True
        self._outputs = set(space_out)

    def forward(self, inputs=None):
        counter = []
        start_ts = time.time()

        def forward_module(module):
            input_modules = self.get_inputs(module)
            if len(input_modules) <= 0:
                # The input module of space
                module.forward(inputs)
            elif len(input_modules) == 1:
                module.forward(inputs=input_modules[0].output)
            else:
                module.forward(inputs=[m.output for m in input_modules])
            counter.append(0)
            return True

        self.traverse(forward_module, direction='forward')
        end_ts = time.time()
        if logger.is_info_enabled():
            logger.debug(f'Forward Space: forwarded {len(counter)} modules in {end_ts - start_ts} seconds.')
        outputs = [output.output for output in self.get_outputs()]
        return outputs

    def compile_and_forward(self, inputs=None, deepcopy=True):
        space = self.compile(deepcopy)
        outputs = space.forward(inputs)
        return space, outputs

    def traverse(self, fn, direction='forward', start_modules=[], discard_isolated_node=True):
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
        if start_modules is None or len(start_modules) <= 0:
            start_modules = fn_inputs(discard_isolated_node=discard_isolated_node)
        for m in start_modules:
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
        if len(self.edges & {(from_module, to_module)}) == 1:
            self.edges.remove((from_module, to_module))

    def disconnect_all(self, module):
        found = set()
        for f, t in self.edges:
            if f == module or t == module:
                found.add((f, t))
        for f, t in found:
            self.edges.remove((f, t))

    def reroute_to(self, old_module, new_module):
        assert isinstance(new_module, (list, ModuleSpace))
        found = set()
        for f, t in self.edges:
            if t == old_module:
                found.add((f, t))

        for f, t in found:
            self.edges.remove((f, t))
            if isinstance(new_module, ModuleSpace):
                self.edges.add((f, new_module))
            else:
                for m in new_module:
                    self.edges.add((f, m))

    def reroute_from(self, old_module, new_module):
        found = set()
        assert isinstance(new_module, (list, ModuleSpace))

        for f, t in self.edges:
            if f == old_module:
                found.add((f, t))

        for f, t in found:
            self.edges.remove((f, t))
            if isinstance(new_module, ModuleSpace):
                self.edges.add((new_module, t))
            else:
                for m in new_module:
                    self.edges.add((m, t))

    def replace_route(self, old_module, new_module):
        self.reroute_to(old_module, new_module)
        self.reroute_from(old_module, new_module)

    def is_isolated_module(self, module):
        assert module is not None
        for from_module, to_module in self.edges:
            if module == from_module or module == to_module:
                return False
        return True

    def get_sub_graph_outputs(self, module_in_subgraph):
        return self.get_sub_graph_end_modules(module_in_subgraph, direction='forward')

    def get_sub_graph_inputs(self, module_in_subgraph):
        return self.get_sub_graph_end_modules(module_in_subgraph, direction='backward')

    def get_sub_graph_end_modules(self, module_in_subgraph, direction='forward'):
        assert isinstance(module_in_subgraph, (ModuleSpace, list))
        if isinstance(module_in_subgraph, list):
            module_in_subgraph = module_in_subgraph[0]

        if direction == 'forward':  # get outputs
            get_upstream = self.get_outputs
            get_downstream = self.get_inputs
        elif direction == 'backward':  # get inputs
            get_upstream = self.get_inputs
            get_downstream = self.get_outputs

        else:
            raise ValueError(f'Not supported direction:{direction}')

        standby = queue.Queue()
        visited = {module_in_subgraph}
        finished = {module_in_subgraph}
        end_modules = set()
        if len(get_upstream(module_in_subgraph)) <= 0:
            end_modules.add(module_in_subgraph)
        for m in get_upstream(module_in_subgraph):
            standby.put(m)
            visited.add(m)
        for m in get_downstream(module_in_subgraph):
            standby.put(m)
            visited.add(m)

        while not standby.empty():
            m_todo = standby.get()
            m_downstream = get_downstream(m_todo)
            ready = True
            for mi in m_downstream:
                if mi not in finished:
                    standby.put(mi)
                    visited.add(mi)
                    ready = False
                    break

            if not ready:
                if len(visited & {m_todo}) > 0:
                    visited.remove(m_todo)
                continue

            finished.add(m_todo)
            m_upstream = get_upstream(m_todo)
            if len(m_upstream) <= 0:
                end_modules.add(m_todo)
                continue

            for m in m_upstream:
                if not m in visited:
                    standby.put(m)
                    visited.add(m)
        return sorted(end_modules, key=lambda m: m.id)

    def set_inputs(self, modules):
        assert modules is not None
        assert isinstance(modules, (ModuleSpace, list))
        if isinstance(modules, ModuleSpace):
            self._inputs = {modules}
        else:
            assert len(modules) > 0
            assert all([isinstance(m, ModuleSpace) for m in modules])
            self._inputs = set(modules)

    def set_outputs(self, modules):
        assert modules is not None
        assert isinstance(modules, (ModuleSpace, list))
        if isinstance(modules, ModuleSpace):
            self._outputs = {modules}
        else:
            assert len(modules) > 0
            assert all([isinstance(m, ModuleSpace) for m in modules])
            self._outputs = set(modules)

    def get_inputs(self, module=None, discard_isolated_node=True):
        inputs = set()
        if module is None:
            if len(self._inputs) > 0:
                return sorted(self._inputs, key=lambda m: m.id)

            for m in self.modules:
                has_in = False
                has_out = False
                for from_module, to_module in self.edges:
                    if m == from_module:
                        has_out = True
                    if m == to_module:
                        has_in = True
                        break
                if not has_in:
                    if has_out or (not discard_isolated_node and m.path == ''):
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

    def get_outputs(self, module=None, discard_isolated_node=True):
        outputs = set()
        if module is None:
            if len(self._outputs) > 0:
                return sorted(self._outputs, key=lambda m: m.id)

            for m in self.modules:
                has_in = False
                has_out = False
                for from_module, to_module in self.edges:
                    if m == from_module:
                        has_out = True
                        break
                    if m == to_module:
                        has_in = True
                if not has_out:
                    if has_in or (not discard_isolated_node and m.path == ''):
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

    def random_sample(self):
        for hp in self.params_iterator:
            hp.random_sample()

    def get_unassigned_params(self, traverse_direction='forward'):
        assignables = []

        def append_params(m):
            ps = m.get_assignable_params()
            for p in ps:
                if p not in assignables:
                    assignables.append(p)
            return True

        self.traverse(append_params, direction=traverse_direction, discard_isolated_node=False)
        return assignables

    def get_assigned_params(self):
        assert self.all_assigned
        return self._assigned_params_stack

    def get_assigned_param_values(self, traverse_direction='forward'):
        ps = self.get_assigned_params()
        return {p.id: p.value for p in ps}

    def get_all_params(self):
        all = list(self.hyper_params)
        return all

    def params_summary(self, only_assignable=True, line_width=60, LR='\n'):
        outputs = []
        outputs.append(f'\n{(line_width + 2) * "-"}')
        if only_assignable:
            params = self.get_assigned_params()
        else:
            params = self.get_all_params()

        for i, hp in enumerate(params):
            outputs.append(
                f'({i}) {hp.alias}:{(line_width - len(str(i) + "() " + hp.alias + str(hp.value))) * " "}{hp.value}')
        outputs.append(f'{(line_width + 2) * "-"}')
        return LR.join(outputs)

    @property
    def signature(self):
        assert self.all_assigned
        labels = [p.label for p in self._assigned_params_stack]
        key = ';'.join(labels)
        md5 = hashlib.md5(key.encode('utf-8')).hexdigest()
        return md5

    @property
    def vectors(self):
        assert self.all_assigned
        vectors = [p.value2numeric(p.value) for p in self._assigned_params_stack]
        return vectors

    def assign_by_vectors(self, vectors):
        i = 0
        for p in self.params_iterator:
            if not p.is_mutable:
                p.random_sample()
                continue
            if i >= len(vectors):
                raise ValueError('`vector` and `space` does not match.')
            p.assign(p.numeric2value(vectors[i]))
            i += 1
        if len(vectors) != i:
            raise ValueError('`vector` and `space` does not match.')

    @property
    def combinations(self):
        count = 1
        for hp in self.hyper_params:
            count *= hp.choice_num
        return count

    def _repr_html_(self):
        html = '''<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th>key</th>
  <th>value</th>
</tr>
</thead>
<tbody>'''
        html += f'''<tr>
  <td>signature</td>
  <td>{self.signature}</td>
</tr>
<tr>
  <td>vectors</td>
  <td>{self.vectors}</td>
</tr>'''
        params = self.get_assigned_params()
        for i, hp in enumerate(params):
            html += f'''<tr>
  <td>{i}-{hp.alias}</td>
  <td>{hp.value}</td>
</tr>
<tr>'''

        html += '''  </tbody>
</table>
</div>'''
        return html


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

    @property
    def config_keys(self):
        raise NotImplementedError

    @property
    def label(self):
        vs = [str(self.__dict__[key]) for key in self.config_keys]
        return f"{self.id}-{'-'.join(vs)}"

    def value2numeric(self, value):
        raise NotImplementedError

    def numeric2value(self, numeric):
        raise NotImplementedError

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
        if self.is_mutable:
            self.space.push_assigned_param(self)
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

    def same_config(self, other):
        if self.__class__ == other.__class__ and self.alias == other.alias:
            for key in self.config_keys:
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        else:
            return False

    def expansion(self, sample_num):
        raise NotImplementedError

    @property
    def choice_num(self):
        if self.is_mutable:
            return self._get_choice_num()
        else:
            return 1

    def _get_choice_num(self):
        raise NotImplementedError


class Int(ParameterSpace):
    def __init__(self, low, high, step=1, random_state=np.random.RandomState(), space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        assert isinstance(low, int) and isinstance(high, int), '`low` and `high` must be a int.'
        assert low < high, '`low` must less than `high`.'
        self.low = low
        self.high = high
        self.step = step
        self.random_state = random_state

    def _random_sample(self):
        value = self.random_state.randint(self.low, self.high)
        if self.step is not None:
            all = np.arange(self.low, self.high + self.step, step=self.step)
            value = all[np.abs(all - value).argmin()]
        return value

    def _check(self, value):
        assert value >= self.low and value <= self.high

    def value2numeric(self, value):
        return value

    def numeric2value(self, numeric):
        return numeric

    @property
    def config_keys(self):
        return ['low', 'high', 'step']

    def expansion(self, sample_num):
        p = self._get_choice_num()
        if sample_num > p or sample_num <= 0:
            sample_num = p
        samples = []
        values = []
        while len(samples) < sample_num:
            v = self._random_sample()
            if v in values:
                continue
            sample = copy.deepcopy(self)
            sample.assign(v)
            samples.append(sample)
            values.append(v)

        return sorted(samples, key=lambda s: s.value)

    def _get_choice_num(self):
        p = self.high - self.low
        if self.step is not None:
            p = len(np.arange(self.low, self.high + self.step, step=self.step)) - 1
        return p


class Real(ParameterSpace):
    def __init__(self, low, high, q=None, prior="uniform", step=0.01, max_expansion=100,
                 random_state=np.random.RandomState(),
                 space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        low = float(low)
        high = float(high)
        assert low < high, '`low` must less than `high`.'
        self.low = low
        self.high = high
        self.q = q
        self.prior = prior
        self.step = float(step)
        self.random_state = random_state
        self.max_expansion = max_expansion

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

        if self.step is not None:
            if self.prior == 'log_uniform':
                all = np.arange(np.exp(self.low), np.exp(self.high) + self.step, step=self.step)
                value = all[np.abs(all - value).argmin()]
                if value > np.exp(self.high):
                    value = np.exp(self.high)
            else:
                all = np.arange(self.low, self.high + self.step, step=self.step)
                value = all[np.abs(all - value).argmin()]
                if value > self.high:
                    value = self.high
        return value

    def _check(self, value):
        if self.prior == "log_uniform":
            assert value >= np.exp(self.low) and value <= np.exp(self.high)
        else:
            assert value >= self.low and value <= self.high

    @property
    def config_keys(self):
        return ['low', 'high', 'q', 'prior', 'step']

    def value2numeric(self, value):
        return value

    def numeric2value(self, numeric):
        return numeric

    def expansion(self, sample_num):
        if sample_num <= 0:
            sample_num = self.max_expansion
        sample_num = min(sample_num, self._get_choice_num())
        values = []
        samples = []
        while len(samples) < sample_num:
            v = self._random_sample()
            if v in values:
                continue
            sample = copy.deepcopy(self)
            sample.assign(v)
            samples.append(sample)
            values.append(v)
        return sorted(samples, key=lambda s: s.value)

    def _get_choice_num(self):
        p = self.max_expansion
        if self.step is not None:
            if self.prior == 'log_uniform':
                p = len(np.arange(np.exp(self.low), np.exp(self.high) + self.step, step=self.step))
            else:
                p = len(np.arange(self.low, self.high + self.step, step=self.step))
        return p


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
        index = self.random_state.choice(range(len(self.options)))
        return self.options[index]

    def _check(self, value):
        assert value in self.options

    @property
    def config_keys(self):
        return ['options']

    def value2numeric(self, value):
        return self.options.index(value)

    def numeric2value(self, numeric):
        return self.options[numeric]

    def expansion(self, sample_num=0):
        samples = []
        for option in self.options:
            sample = copy.deepcopy(self)
            sample.assign(option)
            samples.append(sample)
        return samples

    def _get_choice_num(self):
        return len(self.options)


class MultipleChoice(ParameterSpace):
    def __init__(self, options, num_chosen_most=0, num_chosen_least=1, random_state=np.random.RandomState(),
                 space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        assert isinstance(options, list), '`options` must be a List.'
        assert len(options) >= num_chosen_least, f'`options` contains at least {num_chosen_least} item.'
        self.options = options
        self.num_chosen_most = num_chosen_most
        self.num_chosen_least = num_chosen_least
        self.random_state = random_state

    def _random_sample(self):
        high = self.num_chosen_most
        if high <= 0:
            high = len(self.options)
        indices = self.random_state.choice(range(0, len(self.options)),
                                           self.random_state.randint(self.num_chosen_least, high + 1), False)
        values = [self.options[index] for index in sorted(indices)]
        return values

    def _check(self, value):
        assert isinstance(value, list)
        assert len(value) >= self.num_chosen_least, f'value contains at least {self.num_chosen_least} item.'
        assert (self.num_chosen_most == 0 or self.num_chosen_most <= len(self.options))
        assert all([v in self.options for v in value])

    @property
    def config_keys(self):
        return ['options', 'num_chosen_most', 'num_chosen_least']

    def value2numeric(self, value):
        numeric = int(''.join(['1' if v in value else '0' for v in self.options]), 2)
        return numeric

    def numeric2value(self, numeric):
        bin = np.binary_repr(numeric, len(self.options))

        values = []
        for i in range(len(bin)):
            if bin[i] == '1':
                values.append(self.options[i])
        return values

    def expansion(self, sample_num):
        c = self._get_choice_num()
        if sample_num > c or sample_num <= 0:
            sample_num = c
        values = []
        samples = []
        while len(values) < sample_num:
            v = self._random_sample()
            if v in values:
                continue
            sample = copy.deepcopy(self)
            sample.assign(v)
            samples.append(sample)
            values.append(v)
        return samples

    def _get_choice_num(self):
        return int(combinations(len(self.options), self.num_chosen_most, 1))


class Bool(Choice):
    def __init__(self, random_state=np.random.RandomState(), space=None, name=None):
        Choice.__init__(self, [False, True], random_state, space, name)


class Constant(ParameterSpace):
    def __init__(self, value, space=None, name=None):
        ParameterSpace.__init__(self, space, name)
        self.assign(value)

    @property
    def is_mutable(self):
        return False

    @property
    def config_keys(self):
        return ['_value']


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
            value = self._lambda_fn(**args)
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
                if isinstance(m, ModuleSpace):
                    m.add_parameters(**{name: value})
                elif isinstance(m, ParameterSpace):
                    value.attach(m)
            self.assign(value)

    @property
    def param_dict(self):
        return self._param_dict


class ModuleSpace(HyperNode):
    def __init__(self, space=None, name=None, **hyperparams):
        HyperNode.__init__(self, space, name)
        self._hyper_params = OrderedDict()
        self._is_params_ready = False
        self._is_compiled = False
        self._output = None
        self.space.add_node(self)
        self.add_parameters(**hyperparams)
        self.update()

    def __call__(self, *args, **kwargs):
        assert len(args) > 0
        m = args[0]
        assert m is not None, 'The module to be connected cannot be None.'
        assert isinstance(m, (ModuleSpace, list))
        if isinstance(m, ModuleSpace):
            self.space.connect(m, self)
        else:
            for mi in m:
                assert mi is not None, 'The module to be connected cannot be None.'
                self.space.connect(mi, self)
        return self

    def connect(self, module_or_list):
        if isinstance(module_or_list, ModuleSpace):
            self.space.connect(self, module_or_list)
        elif isinstance(module_or_list, list):
            assert len(module_or_list) > 0, f'module_or_list contains at least 1 Module.'
            assert all([isinstance(m, ModuleSpace) for m in module_or_list]), 'module_or_list can only contain Module.'
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

    def compile(self):
        if not self.is_compiled:
            self._compile()
            self._is_compiled = True

    def _compile(self):
        raise NotImplementedError

    def forward(self, inputs=None):
        self._output = self._forward(inputs)
        return self._output

    def _forward(self, inputs):
        raise NotImplementedError

    def compile_and_forward(self, inputs=None):
        self.compile()
        return self.forward(inputs)

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

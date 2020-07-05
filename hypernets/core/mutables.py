# -*- coding:utf-8 -*-
"""

"""
from collections import OrderedDict


class MutableScope:
    def __init__(self):
        self.reset()
        self.stack = []

    @property
    def current_path(self):
        return '.'.join(self.stack)

    def entry(self, name):
        self.stack.append(name)

    def exit(self):
        self.stack.pop()

    def reset(self):
        self.id_dict = OrderedDict()
        self.name_dict = OrderedDict()

    def register(self, mutable):
        assert isinstance(mutable, Mutable)

        if mutable.name is None:
            mutable.id = self.assign_id(mutable)
            mutable.name = mutable.id
        else:
            if self.name_dict.get(mutable.name) is not None:
                raise ValueError(f'name `{mutable.name}` is duplicate.')
            mutable.id = f'ID_{mutable.name}'

        self.name_dict[mutable.name] = mutable
        self.id_dict[mutable.id] = mutable

    def assign_id(self, mutable):
        prefix = mutable.__class__.__name__
        if mutable.type is not None:
            prefix = mutable.type + '_' + prefix
        i = 1
        while True:
            id = f'{prefix}_{i}'
            if id not in self.id_dict:
                break
            i += 1
        return id

    def get_mutable(self, id):
        return self.id_dict[id]

    def get_mutable_by_name(self, name):
        return self.name_dict[name]


class Mutable(object):
    def __init__(self, scope, name=None):
        assert scope is not None, 'scope cannot be None'
        self.scope = scope
        self.name = name
        self.alias = None
        self.scope.register(self)
        self.path = scope.current_path

    def __repr__(self):
        # if self.alias is not None:
        #     return 'ALIAS:' + self.alias
        # else:
        #     return 'ID:' + self._id
        return self._id

    @property
    def type(self):
        return None

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    def update(self):
        pass

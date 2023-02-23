# -*- coding:utf-8 -*-
"""

"""
import time
from collections import namedtuple

TbNamedEntity = namedtuple('TbNamedEntity', ['entity', 'name', 'dtypes'])

_tool_boxes = []
_tool_boxes_update_at = 0

_tb_transformers = []  # list of TbNamedEntity
_tb_transformers_update_at = 0

_tb_extensions = []  # list of TbNamedEntity
_tb_extensions_update_at = 0


def register_toolbox(tb, pos=None, aliases=None):
    if pos is None:
        _tool_boxes.append((tb, aliases))
    else:
        _tool_boxes.insert(pos, (tb, aliases))

    global _tool_boxes_update_at
    _tool_boxes_update_at = time.time()


def register_transformer(transformer_cls, *, name=None, dtypes=None):
    assert isinstance(transformer_cls, type)
    assert dtypes is None or isinstance(dtypes, (type, list, tuple, set))

    if isinstance(dtypes, type):
        dtypes = [dtypes]
    if dtypes is not None:
        dtypes = set(dtypes)
    if name is None:
        name = transformer_cls.__name__
    _tb_transformers.append(TbNamedEntity(entity=transformer_cls, name=name, dtypes=dtypes))

    global _tb_transformers_update_at
    _tb_transformers_update_at = time.time()


def register_extension(extension, *, name=None, dtypes=None):
    assert dtypes is None or isinstance(dtypes, (type, list, tuple, set))
    assert name is not None or hasattr(extension, '__name__')

    if isinstance(dtypes, type):
        dtypes = [dtypes]
    if dtypes is not None:
        dtypes = set(dtypes)
    if name is None:
        name = extension.__name__

    _tb_extensions.append(TbNamedEntity(entity=extension, name=name, dtypes=dtypes))

    global _tb_extensions_update_at
    _tb_extensions_update_at = time.time()


def get_tool_box(*data):
    assert len(data) > 0
    if len(data) == 1 and isinstance(data[0], str):
        # get toolbox by alias
        for tb, aliases in _tool_boxes:
            if aliases is not None and data[0] in aliases:
                return tb
    else:
        for tb, _ in _tool_boxes:
            if tb.accept(*data):
                return tb

    raise ValueError(f'No toolbox found for your data with types: {[type(x) for x in data]}. '
                     f'Registered tabular toolboxes are {[tb.__name__ for tb, _ in _tool_boxes]}.')


def get_transformers(dtypes):
    """
    Find available transformers for the specified dtypes
    """
    return _filter_entities(_tb_transformers, dtypes)


def get_extensions(dtypes):
    """
    Find available extensions for the specified dtypes
    """
    return _filter_entities(_tb_extensions, dtypes)


def _filter_entities(entity_list, dtypes):
    assert isinstance(dtypes, (list, tuple, set))

    dtypes = set(dtypes)
    found = {}
    unused = []

    # phase 1
    for tf in entity_list:
        if tf.name not in found.keys() \
                and (tf.dtypes is None or dtypes.issubset(tf.dtypes)):
            found[tf.name] = tf.entity
        else:
            unused.append(tf)

    # phase 2
    for tf in unused:
        if tf.name not in found.keys() \
                and len(dtypes.intersection(tf.dtypes)) > 0:
            found[tf.name] = tf.entity

    return found


##################################################################################
# Toolbox MetaClass
#
class ToolboxMeta(type):
    @property
    def qname(cls):
        return f'{cls.__module__}.{cls.__name__}'.replace('.', '_')

    @property
    def transformers(cls):
        qn = cls.qname
        attr_tfs = f'transformers_{qn}_'
        attr_tfs_update_at = f'transformers_update_at_{qn}_'
        tfs = getattr(cls, attr_tfs, None)
        update_at = getattr(cls, attr_tfs_update_at, 0)

        if tfs is None or update_at < _tb_transformers_update_at:
            dtypes = getattr(cls, 'acceptable_types', None)
            tfs = get_transformers(dtypes) if dtypes is not None else {}
            setattr(cls, attr_tfs, tfs)
            setattr(cls, attr_tfs_update_at, time.time())

        return tfs


##################################################################################
# annotations
#

def tb_transformer(dtypes=None, name=None):
    assert dtypes is None or isinstance(dtypes, (type, list, tuple))

    def decorate(cls):
        register_transformer(cls, dtypes=dtypes, name=name)
        return cls

    return decorate


def tb_extension(dtypes, name=None):
    assert dtypes is None or isinstance(dtypes, (type, list, tuple))

    def decorate(ext):
        register_extension(ext, dtypes=dtypes, name=name)
        return ext

    return decorate

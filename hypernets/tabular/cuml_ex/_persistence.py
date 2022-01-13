# -*- coding:utf-8 -*-
"""

"""

import cudf
import cupy

from ..persistence import ParquetPersistence

_my_cached_types = (cudf.DataFrame, cudf.Series, cupy.ndarray)

_META_CUML_KEY = b'cuml_type'


class CumlParquetPersistence(ParquetPersistence):
    acceptable_types = ParquetPersistence.acceptable_types + _my_cached_types

    def store(self, data, path, *, filesystem=None, **kwargs):
        assert isinstance(data, self.acceptable_types)

        metadata = {}
        if isinstance(data, _my_cached_types):
            from . import CumlToolBox
            data, = CumlToolBox.to_local(data)
            metadata[_META_CUML_KEY] = type(data).__name__.encode()

        return super().store(data, path, filesystem=filesystem, metadata=metadata, **kwargs)

    def load(self, path, *, filesystem=None, return_metadata=False, **kwargs):
        data, metadata = super().load(path, filesystem=filesystem, return_metadata=True, **kwargs)

        if metadata is not None and metadata.get(_META_CUML_KEY, None) is not None:
            from . import CumlToolBox
            data, = CumlToolBox.from_local(data)

        if return_metadata:
            return data, metadata
        else:
            return data

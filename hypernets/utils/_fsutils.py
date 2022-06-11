# -*- coding:utf-8 -*-
"""

"""

import json
import os
import tempfile

import fsspec
from fsspec.implementations.local import LocalFileSystem as FsSpecLocalFileSystem

from hypernets.conf import Configurable, configure, Unicode
from . import logging, is_os_windows, Version

logger = logging.get_logger(__name__)


@configure()
class StorageCfg(Configurable):
    kind = Unicode('file',
                   help='storage kind (fsspec protocol). "file" and "s3" were tested.'
                   ).tag(config=True)
    root = Unicode('',
                   help='storage root path.'
                   ).tag(config=True)
    options = Unicode('',
                      help='storage options, json string. see fsspec from more details.'
                      ).tag(config=True)
    local_root = Unicode(os.path.join(tempfile.gettempdir(), 'cache'),
                         help='local root path, used as temp.'
                         ).tag(config=True)


class FileSystemAdapter(object):
    def __init__(self, remote_root, local_root, remote_sep):
        super(FileSystemAdapter, self).__init__()

        # self.fs = fs
        self.remote_root = remote_root
        self.local_root = local_root
        self.remote_sep = remote_sep

    def _inner_to_rpath(self, rpath):
        if rpath.startswith(self.remote_root):
            return rpath

        return self.remote_root.rstrip(self.remote_sep) + self.remote_sep + rpath.lstrip(self.remote_sep)

    def to_rpath(self, rpath):
        # return os.path.join(remote_root, rpath)
        assert rpath

        if isinstance(rpath, str):
            return self._inner_to_rpath(rpath)
        elif isinstance(rpath, (list, tuple)):
            return [self._inner_to_rpath(p) for p in rpath]
        elif isinstance(rpath, os.DirEntry):
            assert rpath.path.startswith(self.remote_root)
            return rpath
        else:
            logger.warn(f'Unexpected rpath type: {type(rpath)}, rpath: {rpath}')
            return rpath

    def to_lpath(self, lpath):
        assert lpath
        if lpath.startswith(self.local_root):
            return lpath

        return os.path.join(self.local_root, lpath)

    def strip_rpath(self, rpath, align_with):
        if isinstance(align_with, os.DirEntry):
            assert align_with.path.startswith(self.remote_root)
            return rpath
        elif align_with.startswith(self.remote_root):
            # if align_with.startswith(remote_sep) and rpath.startswith(align_with.lstrip(remote_sep)):
            #     rpath = remote_sep + rpath
            return rpath

        n = rpath.find(self.to_rpath(align_with).lstrip(self.remote_sep))
        if n == 0 or n == 1:
            rpath = rpath[len(self.remote_root.lstrip(self.remote_sep)):]
            rpath = rpath[rpath.find(align_with):]

        return rpath

    def handle_find(self, result, rpath, *args, **kwargs):
        if kwargs.get('detail'):
            if isinstance(result, dict):
                # fixed = dict([(k, {**v, 'name': strip_rpath(v['name'], rpath)}
                #                ) for k, v in result.items()])
                fixed = {}
                for k, v in result.items():
                    key = self.strip_rpath(k, rpath)
                    value = {**v, 'name': self.strip_rpath(v['name'], rpath)}
                    if 'Key' in v.keys():
                        value['Key'] = self.strip_rpath(v['Key'], rpath)
                    fixed[key] = value
            elif isinstance(result, (list, tuple)):
                fixed = [{**v, 'name': self.strip_rpath(v['name'], rpath)}
                         for v in result]
            else:
                assert False, f'Unexpected result type: {type(result)}'
        else:
            assert isinstance(result, (list, tuple))
            fixed = [self.strip_rpath(path, rpath) for path in result]
        return fixed

    def handle_glob(self, result, rpath, *args, **kwargs):
        assert isinstance(result, (list, tuple))
        n = rpath.find('*')
        if n >= 0:
            fixed = [self.strip_rpath(path, rpath[:n]) for path in result]
        else:
            fixed = [self.strip_rpath(path, rpath) for path in result]
        return fixed

    def handle_info(self, result, rpath, *args, **kwargs):
        assert isinstance(result, dict)
        fixed = {**result, 'name': self.strip_rpath(result['name'], rpath)}
        return fixed

    def handle_walk(self, result, rpath, *args, **kwargs):
        assert type(result).__name__ == 'generator'
        for root, dirs, files in result:
            yield self.strip_rpath(root, rpath), dirs, files

    def fix_r(self, fn, post_handler):
        def execute(rpath, *args, **kwargs):
            result = fn(self.to_rpath(rpath), *args, **kwargs)
            if post_handler:
                result = post_handler(result, rpath, *args, **kwargs)

            # print('-' * 20, fn.__name__, rpath, args, kwargs, result)
            return result

        return execute

    def fix_rr(self, fn, post_handler):
        def execute(rpath1, rpath2, *args, **kwargs):
            return fn(self.to_rpath(rpath1),
                      self.to_rpath(rpath2),
                      *args, **kwargs)

        return execute

    def fix_lr(self, fn, post_handler):
        def execute(lpath, rpath, *args, **kwargs):
            return fn(self.to_lpath(lpath),
                      self.to_rpath(rpath),
                      *args, **kwargs)

        return execute

    def fix_rl(self, fn, post_handler):
        def execute(rpath, lpath, *args, **kwargs):
            return fn(self.to_rpath(rpath),
                      self.to_lpath(lpath),
                      *args, **kwargs)

        return execute

    # functions with one remote path only
    @property
    def fn_r(self):
        return ['cat', 'cat_file', 'checksum', 'created',
                'delete', 'disk_usage', 'du',
                'exists', 'expand_path', 'find', 'glob', 'head',
                'info', 'invalidate_cache', 'isdir', 'isfile',
                'listdir', 'ls',
                'makedir', 'makedirs', 'mkdir', 'mkdirs', 'modified',
                'open', 'pipe', 'pipe_file',
                'rm', 'rm_file', 'rmdir',
                'sign', 'size', 'stat',
                'tail', 'touch', 'ukey', 'walk']

    # functions with remote-remote path pair
    @property
    def fn_rr(self):
        return ['copy', 'cp', 'move', 'mv', 'rename', ]

    # functions with remote-local path pair
    @property
    def fn_rl(self):
        return ['download', 'get', 'get_file', ]

    # functions with local-remote path pair
    @property
    def fn_lr(self):
        return ['put', 'put_file', 'upload']

    # functions without any path
    @property
    def fn_unhandled(self):
        return ['clear_instance_cache', 'current',
                'end_transaction', 'from_json', 'get_mapper',
                'read_block', 'start_transaction', 'to_json']

    # functions to handle result
    @property
    def fn_post_process(self):
        return {'ls': self.handle_find,
                'find': self.handle_find,
                'glob': self.handle_glob,
                'info': self.handle_info,
                'walk': self.handle_walk,
                }

    @property
    def fn_fix_pairs(self):
        return [(self.fn_r, self.fix_r),
                (self.fn_rr, self.fix_rr),
                (self.fn_lr, self.fix_lr),
                (self.fn_rl, self.fix_rl)]

    def __call__(self, fs, *args, **kwargs):
        if hasattr(fs, 'hyn_adapted_'):
            for fns, fix in self.fn_fix_pairs:
                for fn in fns:
                    # assert hasattr(fs, fn), f'fn:{fn}'
                    if not hasattr(fs, fn):
                        continue

                    original_fn = f'_orig_{fn}_'
                    assert hasattr(fs, original_fn)
            assert hasattr(fs, 'remote_root_')
            assert hasattr(fs, 'local_root_')
            return fs

        # decorate listed functions
        post_processes = self.fn_post_process
        for fns, fix in self.fn_fix_pairs:
            for fn in fns:
                # assert hasattr(fs, fn), f'fn:{fn}'
                if not hasattr(fs, fn):
                    continue

                original_fn = f'_orig_{fn}_'
                assert not hasattr(fs, original_fn)

                f = getattr(fs, fn)
                setattr(fs, original_fn, f)
                setattr(fs, fn, fix(f, post_processes.get(fn)))

        # decorate '__reduce__' (pickle compatible)
        f = getattr(fs, '__reduce__')
        setattr(fs, '_orig__reduce__', f)
        setattr(fs, '__reduce__', _fs_reduce)

        # mark adapted
        setattr(fs, 'hyn_adapted_', 1)
        setattr(fs, 'remote_root_', self.remote_root)
        setattr(fs, 'local_root_', self.local_root)

        return fs


class WindowsFileSystemAdapter(FileSystemAdapter):
    def __init__(self, remote_root, local_root, remote_sep):
        remote_root = remote_root.replace('\\', '/')
        local_root = local_root.replace('\\', '/')
        super(WindowsFileSystemAdapter, self).__init__(remote_root, local_root, '/')

    def _inner_to_rpath(self, rpath):
        if rpath.find(':') > 0:
            return rpath
        else:
            return super()._inner_to_rpath(rpath)


class S3FileSystemAdapter(FileSystemAdapter):
    def __init__(self, remote_root, local_root, remote_sep):
        remote_root = remote_root.lstrip(remote_sep)

        super().__init__(remote_root, local_root, remote_sep)

    def handle_private_ls(self, result, rpath, *args, **kwargs):
        assert isinstance(result, (list, tuple))
        fixed = [{**v,
                  'name': self.strip_rpath(v['name'], rpath),
                  'Key': self.strip_rpath(v['Key'], rpath),
                  }
                 for v in result]
        return fixed

    @property
    def fn_r(self):
        import s3fs

        if Version(s3fs.__version__) < Version('0.5.0'):
            return super().fn_r + ['_ls']
        else:
            return super().fn_r

    @property
    def fn_post_process(self):
        return {**super().fn_post_process, '_ls': self.handle_private_ls}


# pyarrow replace fsspec.LocalFileSystem with pyarrow.LocalFileSystem with hard coding, so we declare another
class AdaptedLocalFileSystem(FsSpecLocalFileSystem):
    pass


def get_filesystem(fs_type, fs_root, fs_options) -> fsspec.AbstractFileSystem:
    if fs_type == 'file':
        fs = AdaptedLocalFileSystem()
    elif fs_options is None or len(fs_options) == 0:
        fs = fsspec.filesystem(fs_type, skip_instance_cache=True)
    else:
        try:
            parsed = json.loads(fs_options)
        except json.JSONDecodeError as e:
            msg = 'Failed to parse storage options as json, ' \
                  f'current settings:\n{fs_options}'
            raise Exception(msg) from e

        if not isinstance(parsed, dict):
            msg = 'Storage options should be json dictionary, ' \
                  f'current settings:\n{fs_options}'
            raise Exception(msg)

        fs = fsspec.filesystem(fs_type, skip_instance_cache=True, **parsed)

    if type(fs).__name__.lower().find('local') >= 0:
        if fs_root is None or fs_root == '':
            fs_root = os.path.join(tempfile.gettempdir(), 'workdir')
            # logger.info(f'use {fs_root} as working directory.')

        remote_root = os.path.abspath(os.path.expanduser(fs_root))
        try:
            if not fs.exists(remote_root):
                fs.mkdirs(remote_root, exist_ok=True)
            else:
                with tempfile.TemporaryFile(prefix='hyn_', dir=remote_root) as t:
                    t.write(b'.')
        except PermissionError as e:
            logger.warn(f'{type(e).__name__}: working directory "{remote_root}"')
            logger.warn(e)

        local_root = remote_root
        is_local = True
    else:
        remote_root = fs_root if fs_root else '/tmp'
        if not fs.exists(remote_root):
            fs.mkdirs(remote_root, exist_ok=True)

        local_root = StorageCfg.local_root
        local_root = os.path.abspath(os.path.expanduser(local_root))
        os.makedirs(local_root, exist_ok=True)
        is_local = False

    # return fix_filesystem(fs, remote_root, local_root)
    if type(fs).__name__ == 'S3FileSystem':
        return S3FileSystemAdapter(remote_root, local_root, fs.sep)(fs)
    elif is_os_windows:
        return WindowsFileSystemAdapter(remote_root, local_root, os.path.sep)(fs)
    else:
        return FileSystemAdapter(remote_root, local_root, os.path.sep if is_local else fs.sep)(fs)


def _fs_reduce(*args):
    return get_filesystem, (StorageCfg.kind if len(StorageCfg.kind) > 0 else None,
                            StorageCfg.root if len(StorageCfg.root) > 0 else None,
                            StorageCfg.options if len(StorageCfg.options) > 0 else None)


filesystem = _fs_reduce()[0](*_fs_reduce()[1])

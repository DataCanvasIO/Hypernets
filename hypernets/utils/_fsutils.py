# -*- coding:utf-8 -*-
"""

"""

import json
import os

import fsspec

from .common import config


def get_filesystem() -> fsspec.AbstractFileSystem:
    fs_type = config('storage_type', 'file')
    fs_options = config('storage_options', None)

    if fs_options is None or len(fs_options) == 0:
        fs = fsspec.filesystem(fs_type)
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

        fs = fsspec.filesystem(fs_type, **parsed)

    if is_local(fs):
        remote_root = config('storage_root', os.path.abspath('./workdir'))
        if not fs.exists(remote_root):
            fs.mkdirs(remote_root, exist_ok=True)

        local_root = remote_root
    else:
        remote_root = config('storage_root', '/')
        if not fs.exists(remote_root):
            fs.mkdirs(remote_root, exist_ok=True)

        local_root = config('storage_localroot', os.path.abspath('./workdir/cache'))
        os.makedirs(local_root, exist_ok=True)

    return fix_filesystem(fs, remote_root, local_root)


def is_local(fs):
    return fs.__class__.__name__.lower().find('local') >= 0


def fix_filesystem(fs, remote_root, local_root):
    remote_sep = os.path.sep if is_local(fs) else '/'

    def to_rpath(rpath):
        # return os.path.join(remote_root, rpath)
        assert rpath
        if rpath.startswith(remote_root):
            return rpath

        return remote_root.rstrip(remote_sep) + remote_sep + rpath.lstrip(remote_sep)

    def to_lpath(lpath):
        assert lpath
        if lpath.startswith(local_root):
            return lpath

        return os.path.join(local_root, lpath)

    def fix_r(fn):
        def execute(rpath, *args, **kwargs):
            return fn(to_rpath(rpath), *args, **kwargs)

        return execute

    def fix_rr(fn):
        def execute(rpath1, rpath2, *args, **kwargs):
            return fn(to_rpath(rpath1),
                      to_rpath(rpath2),
                      *args, **kwargs)

        return execute

    def fix_lr(fn):
        def execute(lpath, rpath, *args, **kwargs):
            return fn(to_lpath(lpath),
                      to_rpath(rpath),
                      *args, **kwargs)

        return execute

    def fix_rl(fn):
        def execute(rpath, lpath, *args, **kwargs):
            return fn(to_rpath(rpath),
                      to_lpath(lpath),
                      *args, **kwargs)

        return execute

    # functions with one remote path only
    fn_r = ['cat', 'cat_file', 'checksum', 'created',
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
    fn_rr = ['copy', 'cp', 'move', 'mv', 'rename', ]

    # functions with remote-local path pair
    fn_rl = ['download', 'get', 'get_file', ]

    # functions with local-remote path pair
    fn_lr = ['put', 'put_file', 'upload']

    # functions without any path
    fn_unhandled = ['clear_instance_cache', 'current',
                    'end_transaction', 'from_json', 'get_mapper',
                    'read_block', 'start_transaction', 'to_json']

    fn_fix = [(fn_r, fix_r),
              (fn_rr, fix_rr),
              (fn_lr, fix_lr),
              (fn_rl, fix_rl)]

    for fns, fix in fn_fix:
        for fn in fns:
            f = getattr(fs, fn)
            original_fn = f'_orig_{fn}_'
            assert not hasattr(fs, original_fn)
            setattr(fs, original_fn, f)
            setattr(fs, fn, fix(f))

    return fs


filesystem = get_filesystem()

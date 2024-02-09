# -*- coding:utf-8 -*-

from .local_process import LocalProcess

try:
    from .grpc_process import GrpcProcess
except ImportError:
    pass
except:
    from hypernets.utils import logging
    import sys

    logger = logging.get_logger(__name__)
    logger.warning('Failed to load GrpcProcess', exc_info=sys.exc_info())

try:
    from .ssh_process import SshProcess
except ImportError:
    pass

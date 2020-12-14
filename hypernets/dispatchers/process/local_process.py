# -*- coding:utf-8 -*-

import subprocess
import sys
from multiprocessing import Process, Value as PValue

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class LocalProcess(Process):
    def __init__(self, cmd, in_file, out_file, err_file, environment=None):
        super(LocalProcess, self).__init__()
        self.cmd = cmd
        self.in_file = in_file
        self.out_file = out_file
        self.err_file = err_file
        self.environment = environment
        self._exit_code = PValue('i', -1)

    def run(self):
        if logger.is_info_enabled():
            logger.info(f'[{self.name}] [CMD] {self.cmd}, out={self.out_file}, err={self.err_file}')

        try:
            if self.out_file and self.err_file:
                with open(self.out_file, 'wb', buffering=0)as o, open(self.err_file, 'wb', buffering=0) as e:
                    p = subprocess.run(self.cmd.split(' '),
                                       shell=False,
                                       stdin=subprocess.DEVNULL,
                                       stdout=o,
                                       stderr=e)
                    code = p.returncode
            else:
                p = subprocess.run(self.cmd.split(' '),
                                   shell=False,
                                   stdin=subprocess.DEVNULL,
                                   stdout=sys.stdout,
                                   stderr=sys.stderr)
                code = p.returncode
        except KeyboardInterrupt:
            code = 137

        if logger.is_info_enabled():
            logger.info(f'[{self.name}] [CMD] {self.cmd} done with {code}')

        self._exit_code.value = code

    @property
    def exitcode(self):
        code = self._exit_code.value
        return code if code >= 0 else None

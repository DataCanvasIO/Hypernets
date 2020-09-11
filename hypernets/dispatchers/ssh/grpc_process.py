# -*- coding:utf-8 -*-

import sys
from multiprocessing import Process, Value as PValue

from ..grpc.process_broker_client import ProcessBrokerClient


class GrpcProcess(Process):
    def __init__(self, grpc_broker, cmd, in_file, out_file, err_file, environment=None):
        super(GrpcProcess, self).__init__()

        self.grpc_broker = grpc_broker
        self.cmd = cmd
        self.in_file = in_file
        self.out_file = out_file
        self.err_file = err_file
        self.environment = environment
        self._exit_code = PValue('i', -1)

    def run(self):
        print(f'[GRPC {self.grpc_broker}] {self.cmd}, out={self.out_file}, err={self.err_file}')

        client = ProcessBrokerClient(self.grpc_broker)
        buffer_size = 16
        if self.out_file and self.err_file:
            with open(self.out_file, 'wb', buffering=0)as o, open(self.err_file, 'wb', buffering=0) as e:
                code = client.run(self.cmd.split(' '), stdout=o, stderr=e, buffer_size=buffer_size)
        else:
            code = client.run(self.cmd.split(' '), stdout=sys.stdout, stderr=sys.stderr, buffer_size=buffer_size)

        print(f'[GRPC {self.grpc_broker}] {self.cmd} done with {code}')
        self._exit_code.value = code

    @property
    def exitcode(self):
        code = self._exit_code.value
        return code if code >= 0 else None

# -*- coding:utf-8 -*-

from multiprocessing import Process, Value as PValue
import subprocess


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
        with open(self.out_file, 'wb')as o, open(self.err_file, 'wb') as e:
            p = subprocess.run(self.cmd.split(' '),
                               shell=True,
                               stdin=None,
                               stdout=o,
                               stderr=e)
            code = p.returncode

        self._exit_code.value = code

    @property
    def exitcode(self):
        code = self._exit_code.value
        return code if code >= 0 else None


if __name__ == '__main__':
    p = LocalProcess('ls', None, 'my.out', 'my.err')
    p.start()
    p.join()
    print(p.cmd, 'exit with', p.exitcode)

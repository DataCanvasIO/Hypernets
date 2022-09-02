import json
import os
from pathlib import Path
from typing import Dict, Optional
import copy
import psutil
from tornado import ioloop

from hypernets import __version__ as hyn_version
from hypernets.hyperctl.batch import Batch, _ShellJob
from hypernets.hyperctl.executor import create_executor_manager
from hypernets.hyperctl.scheduler import JobScheduler
from hypernets.hyperctl.server import create_batch_manage_webapp
from hypernets.hyperctl.utils import load_json, http_portal
from hypernets.utils import logging


logger = logging.getLogger(__name__)


class BatchApplication:

    def __init__(self, batch: Batch,
                 server_host="localhost",
                 server_port=8060,
                 scheduler_exit_on_finish=True,
                 scheduler_interval=5000,
                 scheduler_callbacks=None,
                 scheduler_signal_file=None,
                 independent_tmp=True,
                 backend_conf=None,
                 **kwargs):

        self.batch = batch
        self.independent_tmp = independent_tmp  # allocate tmp for every job

        self.job_scheduler: JobScheduler = self._create_scheduler(backend_conf,
                                                                  server_host, server_port,
                                                                  scheduler_exit_on_finish,
                                                                  scheduler_interval,
                                                                  scheduler_callbacks,
                                                                  scheduler_signal_file,
                                                                  independent_tmp)
        # create web app
        self.web_app = self._create_web_app(server_host, server_port, batch)

        self._http_server = None

    def _create_web_app(self, server_host, server_port, batch):
        return create_batch_manage_webapp(server_host, server_port, batch, self.job_scheduler)

    def _create_scheduler(self, backend_conf, server_host, server_port, scheduler_exit_on_finish,
                          scheduler_interval, scheduler_callbacks, scheduler_signal_file, independent_tmp):

        executor_manager = create_executor_manager(backend_conf, server_host, server_port)
        return JobScheduler(batch=self.batch, exit_on_finish=scheduler_exit_on_finish,
                            interval=scheduler_interval, executor_manager=executor_manager,
                            callbacks=scheduler_callbacks,
                            signal_file=scheduler_signal_file, independent_tmp=independent_tmp)

    def start(self):
        logger.info(f"batch name: {self.batch.name}")
        logger.info(f"batch data dir: {self.batch.data_dir_path.absolute()}")

        # prepare batch data dir
        if self.batch.data_dir_path.exists():
            logger.info(f"batch {self.batch.name} already exists, try to recovery state...")
            for job in self.batch.jobs:
                job: _ShellJob = job
                j_status = self.batch.get_persisted_job_status(job.name)
                job.set_status(j_status)
            logger.info(self.batch.summary())
        else:
            os.makedirs(self.batch.data_dir_path, exist_ok=True)

        # write batch config
        batch_config_file_path = self.batch.config_file_path()
        batch_as_config = self.to_config()
        with open(batch_config_file_path, 'w', newline='\n') as f:
            json.dump(batch_as_config, f, indent=4)
        logger.debug(f"write config to file {batch_config_file_path}")

        # write pid file
        with open(self.batch.pid_file_path(), 'w', newline='\n') as f:
            f.write(str(os.getpid()))

        # start web server
        self._http_server = self.web_app.listen(self.server_port)
        # self._http_server.start()
        server_portal = http_portal(self.server_host, self.server_port)
        logger.info(f"api server is ready to run at: {server_portal}")

        # start scheduler
        self.job_scheduler.start()

    def stop(self):
        if self.job_scheduler is None:
            raise RuntimeError("job_scheduler is None, maybe not started yet")
        if self._http_server is None:
            raise RuntimeError("_http_server is None, maybe not started yet")

        self.job_scheduler.stop()
        self._http_server.stop()
        logger.info(f"stopped api server")

    def to_config(self):
        jobs_config = []
        for job in self.batch.jobs:
            jobs_config.append(job.to_config())

        batch_as_config = {
            "jobs": jobs_config,
            "name": self.batch.name,
            "job_command": self.batch.job_command,
            "server": {
                "host": self.server_host,
                "port": self.server_port
            },
            "scheduler": {
                "interval": self.job_scheduler.interval,
                "exit_on_finish": self.job_scheduler.exit_on_finish
            },
            "version": hyn_version
        }
        return batch_as_config

    def summary_batch(self):
        batch = self.batch

        batch_summary = batch.summary()
        batch_summary['portal'] = http_portal(self.server_host, self.server_port)
        return batch_summary

    @staticmethod
    def load(batch_spec_dict: Dict, batch_data_dir):

        batch_spec_dict = copy.copy(batch_spec_dict)

        def flat_args(config_key: str):
            if config_key in batch_spec_dict:
                sub_config: Dict = batch_spec_dict.pop(config_key)
                sub_init_kwargs = {f"{config_key}_{k}": v for k, v in sub_config.items()}
                batch_spec_dict.update(sub_init_kwargs)

        batch_name = batch_spec_dict.pop('name')
        job_command = batch_spec_dict.pop('job_command')
        jobs_dict = batch_spec_dict.pop('jobs')

        batch = Batch(name=batch_name, data_dir=batch_data_dir, job_command=job_command)
        for job_dict in jobs_dict:
            # job.set_status(batch.get_job_status(job_name=job.name))
            batch.add_job(**job_dict)
            # batch.add_job(**job_dict)

        flat_args("server")
        flat_args("scheduler")

        backend_conf = batch_spec_dict.get('backend')

        # web application
        app = BatchApplication(batch, backend_conf=backend_conf, **batch_spec_dict)

        return app

    @property
    def server_host(self):
        return self.web_app.host

    @property
    def server_port(self):
        return self.web_app.port

from collections import OrderedDict

import numpy as np

from hypernets.hyperctl.batch import Batch, _ShellJob
from hypernets.utils import logging as hyn_logging

logger = hyn_logging.getLogger(__name__)


class BatchCallback:

    def on_start(self, batch):
        pass

    def on_job_start(self, batch, job, executor):
        pass

    def on_job_succeed(self, batch, job, executor, elapsed: float):
        pass

    def on_job_failed(self, batch, job, executor, elapsed: float):
        """Job ran failed """
        pass

    def on_job_break(self, batch, job, exception):
        """ Job failed before running"""
        pass

    def on_finish(self, batch, elapsed: float):
        """Batch finished"""
        pass


class ConsoleCallback(BatchCallback):

    def on_start(self, batch):
        print("on_start")

    def on_job_start(self, batch, job, executor):
        print("on_job_start")

    def on_job_succeed(self, batch, job, executor, elapsed: float):
        print("on_job_succeed")

    def on_job_failed(self, batch, job, executor, elapsed: float):
        print("on_job_failed")

    def on_job_break(self, batch, job, exception):
        print("on_job_break")

    def on_finish(self, batch, elapsed: float):
        print("on_finish")


class VisDOMCallback(BatchCallback):

    def __init__(self):
        self._elapsed = OrderedDict()
        self._n_succeed = 0
        self._n_failed = 0
        self._n_running = 0
        self._n_not_start = 0
        self._n_other = 0

        self.node_jobs = {}

        self.session_ = None

    def _ensure_session(self):
        import visdom
        if self.session_ is None or not self.session_.check_connection(timeout_seconds=3):  # lazy singleton
            self.session_ = visdom.Visdom()
        return self.session_

    def _update_chart(self):

        summary_dict = {
            _ShellJob.STATUS_SUCCEED: self._n_succeed,
            _ShellJob.STATUS_RUNNING: self._n_running,
            _ShellJob.STATUS_FAILED: self._n_failed,
            _ShellJob.STATUS_INIT: self._n_not_start,
            "other": self._n_other
        }

        vis = self._ensure_session()

        vis.pie(
            win='summary_task',
            X=np.array(list(summary_dict.values())),
            opts=dict(legend=list(summary_dict.keys()))
        )

        # only 1 items can not display
        elapsed_array = np.array(list(self._elapsed.values()))

        if np.squeeze(elapsed_array).ndim in [1, 2]:
            vis.bar(win='summary_task_elapsed', X=elapsed_array, opts=dict(rownames=list(self._elapsed.keys())))

        # job on each machine
        node_jobs_array = np.array(list(self.node_jobs.values()))
        if np.squeeze(node_jobs_array).ndim in [1, 2]:
            vis.bar(win='summary_node_jos', X=node_jobs_array, opts=dict(rownames=list(self.node_jobs.keys())))

        # vis.close()

    def on_start(self, batch: Batch):
        # init job time cost
        for job in batch.jobs:
            if job.status == _ShellJob.STATUS_SUCCEED:
                job_state = job.state_data()
                self._update_node_jobs(job_state['ext'])

                self._elapsed[job.name] = job_state['elapsed']

        # init job summary
        for job in batch.jobs:
            job_status = batch.get_job_status(job.name)
            if job_status == _ShellJob.STATUS_SUCCEED:
                self._n_succeed = self._n_succeed + 1
            elif job_status == _ShellJob.STATUS_RUNNING:
                self._n_running = self._n_running + 1
            elif job_status == _ShellJob.STATUS_FAILED:
                self._n_failed = self._n_failed + 1
            elif job_status == _ShellJob.STATUS_INIT:
                self._n_not_start = self._n_not_start + 1
            else:
                self._n_other = self._n_other + 1

        self._update_chart()

    def _update_node_jobs(self, job_ext):
        if job_ext.get('backend_type') == 'remote':
            hostname = job_ext.get('hostname')
            self.node_jobs[hostname] = self.node_jobs.get(hostname, 0) + 1

    def on_job_start(self, batch, job, executor):
        self._n_running = self._n_running + 1

        self._update_chart()

    def on_job_succeed(self, batch: Batch, job, executor, elapsed: float):
        # increment cost
        self._elapsed[job.name] = elapsed

        # update items
        self._n_succeed = self._n_succeed + 1
        self._n_not_start = self._n_not_start - 1

        self._update_node_jobs(job.ext)

        self._n_running = self._n_running - 1

        # re-redner
        self._update_chart()

    def on_job_break(self,  batch, job, exception):
        self.on_job_failed(batch, job, None, -1)

    def on_job_failed(self, batch, job, executor, elapsed: float):
        self._n_failed = self._n_failed + 1
        self._n_not_start = self._n_not_start - 1

        self._n_running = self._n_running - 1

        self._update_chart()

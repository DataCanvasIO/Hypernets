from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd

from hypernets.hyperctl.batch import Batch, _ShellJob
from hypernets.utils import logging as hyn_logging
from hypernets.hyperctl import consts
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

    def __init__(self, n_tail_jobs=100, elapsed_cut_bins=10, datetime_qcut_bins=10):
        self.n_tail_jobs = n_tail_jobs

        self.elapsed_cut_bins = elapsed_cut_bins
        self.datetime_qcut_bins = datetime_qcut_bins

        self._session_ = None
        self._job_db_ = None

    def _ensure_session(self):
        import visdom
        if self._session_ is None or not self._session_.check_connection(timeout_seconds=3):  # lazy singleton
            self._session_ = visdom.Visdom()
        return self._session_

    def _update_chart(self):

        vis = self._ensure_session()

        assert self._job_db_ is not None
        df_job_db = self._job_db_.copy()

        # plot status summary
        status_summary_dict = df_job_db['status'].value_counts().to_dict()
        status_summary_dict.setdefault(_ShellJob.STATUS_SUCCEED, 0)
        status_summary_dict.setdefault(_ShellJob.STATUS_INIT, 0)
        status_summary_dict.setdefault(_ShellJob.STATUS_FAILED, 0)
        status_summary_dict.setdefault(_ShellJob.STATUS_RUNNING, 0)

        vis.pie(
            win='job_status_summary',
            X=np.array(list(status_summary_dict.values())),
            opts=dict(title="Job status summary", legend=list(status_summary_dict.keys()))
        )

        # plot last {self.n_tail_jobs} jobs elapsed
        last_succeed_job = df_job_db[df_job_db['status'] == _ShellJob.STATUS_SUCCEED]\
            .sort_values(by='end_datetime', ascending=False).head(self.n_tail_jobs)

        if last_succeed_job.shape[0] > 1:  # can not plot if only one row
            vis.bar(win='last_job_elapsed', X=last_succeed_job['elapsed'],
                    opts=dict(title=f"Last {self.n_tail_jobs} succeed jobs", xlabel="Job name", ylabel="Elapsed",
                              rownames=last_succeed_job.index.tolist()))

        # plot elapsed cut summary
        df_job_db_elapsed_not_null = df_job_db[df_job_db['elapsed'].notnull()].copy()
        if df_job_db_elapsed_not_null.shape[0] > 1:
            df_job_db_elapsed_not_null['cut_elapsed'] = pd.cut(df_job_db_elapsed_not_null['elapsed'],
                                                               bins=self.elapsed_cut_bins, precision=2)
            elapsed_cut_dict = {}
            for k, v in df_job_db_elapsed_not_null['cut_elapsed'].value_counts().to_dict().items():
                elapsed_cut_dict[str(k)] = v

            vis.bar(
                win='elapsed_cut_summary',
                X=np.array(list(elapsed_cut_dict.values())),
                opts=dict(title=f"Elapsed cut summary({self.elapsed_cut_bins} bins)", showlegend=False,
                          rownames=list(elapsed_cut_dict.keys()), xlabel="Elapsed range", ylabel="Jobs count")
            )

        # plot elapsed cut box
        df_valid_end_datetime = df_job_db[df_job_db['end_datetime'].notnull()].copy()
        if df_valid_end_datetime.shape[0] >= self.datetime_qcut_bins and df_valid_end_datetime.shape[0] > 1:

            df_valid_end_datetime['end_datetime_qcut'] = pd.qcut(df_valid_end_datetime['end_datetime'].astype('float'),
                                                                 q=self.datetime_qcut_bins,
                                                                 duplicates='drop', precision=2)

            df_box = pd.DataFrame()
            min_len = min([len(_) for _ in df_valid_end_datetime[['end_datetime_qcut', 'elapsed']]
                          .groupby('end_datetime_qcut').groups.values()])

            for k, v in df_valid_end_datetime[['end_datetime_qcut', 'elapsed']]\
                    .groupby('end_datetime_qcut').groups.items():
                df_box[k] = df_valid_end_datetime.loc[v[:min_len]].reset_index()['elapsed']

            def format_timestamp(t):
                return datetime.fromtimestamp(t).strftime('%m-%d %H:%M:%S')

            def format_interval(i):
                return " - ".join((format_timestamp(i.right), format_timestamp(i.left)))
            if df_box.shape[0] > 1:
                vis.boxplot(df_box.values,
                            win='elapsed_boxplot',
                            opts=dict(title=f"Elapsed boxplot({self.datetime_qcut_bins} bins)",
                                      xlabel="Time Range", ylabel="Elapsed", showlegend=False,
                                      legend=[format_interval(c) for c in df_box.columns]))

        # plot job count on each node
        df_job_db_end_node_not_null = df_job_db[df_job_db['node'].notnull()].copy()
        if df_job_db_end_node_not_null.shape[1] > 0:
            node_summary_dict = df_job_db_end_node_not_null['node'].value_counts().to_dict()
            node_jobs_array = list(node_summary_dict.values())
            if len(node_jobs_array) > 1:
                vis.bar(win='summary_node_jobs', X=node_jobs_array, opts=dict(title="succeed jobs of each node",
                                                                              rownames=list(node_summary_dict.keys()),
                                                                              xlabel="Node name",
                                                                              ylabel="Jobs count"))
        # vis.close()

    def on_start(self, batch: Batch):
        job_rows = []
        # init job summary
        for job in batch.jobs:
            job_status = job.status
            job_node = None
            job_elapsed = None
            if job_status == _ShellJob.STATUS_SUCCEED:
                job_state = job.state_data()
                job_ext = job_state['ext']
                job_elapsed = job_state['elapsed']
                if job_ext.get('backend_type') == 'remote':
                    job_node = job_ext.get('hostname')
                else:
                    job_node = consts.HOST_LOCALHOST

            row = (job.name, job_status, job_node, job_elapsed, job.start_datetime, job.end_datetime)
            job_rows.append(row)

        job_db = pd.DataFrame(data=job_rows, columns=['name', 'status', 'node', 'elapsed',
                                                      'start_datetime', 'end_datetime'])

        job_db.set_index('name', inplace=True)

        self._job_db_ = job_db
        self._update_chart()

    @staticmethod
    def _extract_job_state(job):
        if job.status != _ShellJob.STATUS_SUCCEED:
            return None, None
        else:
            job_state = job.state_data()
            job_ext = job_state['ext']
            job_elapsed = job_state['elapsed']
            if job_ext.get('backend_type') == 'remote':
                return job_ext.get('hostname'), job_elapsed
            else:
                return consts.HOST_LOCALHOST, job_elapsed

    def on_job_start(self, batch, job, executor):
        self._update_on_event(job)

    def on_job_succeed(self, batch: Batch, job, executor, elapsed: float):
        self._update_on_event(job)

    def _update_on_event(self, job):
        assert self._job_db_ is not None
        # job should in self._job_db
        self._job_db_.loc[job.name, 'status'] = job.status

        if job.status == _ShellJob.STATUS_SUCCEED:
            job_node, job_elapsed = self._extract_job_state(job)
            self._job_db_.loc[job.name, 'elapsed'] = job_elapsed

            self._job_db_.loc[job.name, 'start_datetime'] = job.start_datetime
            self._job_db_.loc[job.name, 'end_datetime'] = job.end_datetime
            self._job_db_.loc[job.name, 'node'] = job_node

        # re-redner
        self._update_chart()

    def on_job_break(self,  batch, job, exception):
        self._update_on_event(job)

    def on_job_failed(self, batch, job, executor, elapsed: float):
        self._update_on_event(job)

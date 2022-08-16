import json
import logging
import os.path
from hypernets.utils import logging as hyn_logging

logger = hyn_logging.getLogger(__name__)

class BatchCallback:

    def on_start(self, batch):
        pass

    def on_job_start(self, batch, job, executor):
        pass

    def on_job_finish(self, batch, job, executor, elapsed: float):
        pass

    def on_job_break(self, batch, job, executor, elapsed: float):  # TODO
        pass

    def on_finish(self, batch, elapsed: float):
        pass


class ConsoleCallback(BatchCallback):

    def on_start(self, batch):
        print("on_start")

    def on_job_start(self, batch, job, executor):
        print("on_job_start")

    def on_job_finish(self, batch, job, executor, elapsed: float):
        print("on_job_finish")

    def on_job_break(self, batch, job, executor, elapsed: float):
        print("on_job_break")

    def on_finish(self, batch, elapsed: float):
        print("on_finish")


class JobStateDataCallback(BatchCallback):

    def on_job_finish(self, batch, job, executor, elapsed: float):
        job_log = {
            "start_datetime": job.start_time,
            "elapsed": elapsed,
            "end_datetime": job.end_time,
        }

        # check state file
        state_data_path = batch.job_state_data_file_path(job.name)

        if os.path.exists(state_data_path):
            logger.info(f"state data file {state_data_path} already exists will be overwritten ")

        # write state
        with open(batch.job_state_data_file_path(job.name), 'w') as f:
            json.dump(job_log, f)

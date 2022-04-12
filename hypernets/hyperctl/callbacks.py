

class BatchCallback:

    def on_start(self, batch):
        pass

    def on_job_start(self, batch, job, executor):
        pass

    def on_job_finish(self, batch, job, executor, elapsed: float):
        pass

    def on_job_break(self):  # TODO
        pass

    def on_finish(self, batch, elapsed: float):
        pass


class ConsoleCallback(BatchCallback):

    def on_start(self, batch):
        print(batch)

    def on_job_start(self, batch, job, executor):
        pass

    def on_job_finish(self, batch, job, executor, elapsed: float):
        pass

    def on_job_break(self):
        pass

    def on_finish(self, batch, elapsed: float):
        pass

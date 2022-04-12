

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

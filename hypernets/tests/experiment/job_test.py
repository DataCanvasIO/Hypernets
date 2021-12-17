from hypernets.experiment._job import JobGroupControlCLI


class TestJob:

    def test_job(self):
        p = "C:\\Users\\wuhf\\PycharmProjects\\Hypernets\\hypernets\\tests\\experiment\\bankdata_config.yaml"
        JobGroupControlCLI().main("hypergbm", p)

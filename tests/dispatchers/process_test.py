import os


def start_broker(host, port):
    from hypernets.dispatchers import run_broker
    from hypernets.dispatchers.process import LocalProcess

    broker_cmd = f'python -m {run_broker.__name__} --host={host} --port={port}'
    broker = LocalProcess(broker_cmd, None, None, None)
    broker.start()

    return broker


def test_grpc_broker_run():
    try:
        from paramiko import SSHClient, AutoAddPolicy
        package_exists = True
    except:
        package_exists = False
    if not package_exists:
        return

    import tempfile
    from hypernets.dispatchers.process import GrpcProcess
    from hypernets.utils.common import generate_id

    broker_host = '127.0.0.1'
    broker_port = 43218
    broker = start_broker(broker_host, broker_port)

    # run process
    cmd = 'echo 123'
    temp_dir = tempfile.gettempdir()
    test_id = generate_id()
    out_file, err_file = f'{temp_dir}/test_out_{test_id}.out', f'{temp_dir}/test_out_{test_id}.err'
    proc = GrpcProcess(f'{broker_host}:{broker_port}', cmd, None, out_file, err_file)
    proc.run()
    code = proc.exitcode

    with open(out_file, 'r') as f:
        out = f.read()
    with open(err_file, 'r') as f:
        err = f.read()

    # clean down
    os.remove(out_file), os.remove(err_file)
    broker.terminate()  # todo: fix LocalProcess

    # assert
    assert code == 0
    assert out == '123\n'
    assert err.startswith('pid:')

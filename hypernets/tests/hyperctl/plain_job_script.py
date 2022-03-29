from hypernets import hyperctl


def main():
    params = hyperctl.get_job_params()
    assert params
    print(params)


if __name__ == '__main__':
    main()

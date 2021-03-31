import glob
import os
import sys
from traitlets import Unicode
from traitlets.config import Application, Configurable


class Configuration(Application):
    config_dir = Unicode('./conf',
                         help='The file system directory which contains all configuration files'
                         ).tag(config=True)

    aliases = {
        'log-level': 'Application.log_level',
        'config-dir': 'Configuration.config_dir',
    }

    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        aliases_keys = self.aliases.keys()
        argv = [a for a in sys.argv if any([a.startswith(f'--{k}') for k in aliases_keys])]
        super(Configuration, self).initialize(argv)

        if len(self.config_dir) > 0:
            config_dir = os.path.abspath(os.path.expanduser(self.config_dir))
            for f in glob.glob(f'{config_dir}/*.py', recursive=False):
                self.load_config_file(f)


_conf = Configuration()

generate_config_file = _conf.generate_config_file


def configure():
    """
    Annotation utility to configure one configurable class
    """

    def wrapper(c):
        assert issubclass(c, Configurable)
        o = c(parent=_conf)

        if c not in _conf.classes:
            _conf.classes += [c]

        return o

    return wrapper


def configure_and_observe(obj, observe_names, observe_handle):
    """
    Annotation utility to configure one configurable class and observe it (or other configured one)
    """
    assert (obj is None or isinstance(obj, Configurable)) \
           and callable(observe_handle) \
           and isinstance(observe_names, (tuple, list, str))

    def wrapper_and_observe(c):
        assert issubclass(c, Configurable)
        o = c(parent=_conf)

        if c not in _conf.classes:
            _conf.classes += [c]

        names = observe_names if isinstance(observe_names, (tuple, list)) else [observe_names]
        if obj is None:
            o.observe(observe_handle, names)
        else:
            obj.observe(observe_handle, names)

        return o

    return wrapper_and_observe


def observe(obj, names, handle):
    """
    A utility to observe configured object
    """
    assert isinstance(obj, Configurable) and callable(handle) \
           and isinstance(names, (tuple, list, str))

    names = names if isinstance(names, (tuple, list)) else [names]
    obj.observe(handle, names)

    return obj

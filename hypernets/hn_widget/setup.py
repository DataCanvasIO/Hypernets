from __future__ import print_function
from setuptools import setup, find_packages
import os
from os.path import join as pjoin
from distutils import log

from jupyter_packaging import (
    create_cmdclass,
    install_npm,
    ensure_targets,
    combine_commands,
    get_version,
)

# https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Custom.html

here = os.path.dirname(os.path.abspath(__file__))

log.set_verbosity(log.DEBUG)
log.info('setup.py entered')
log.info('$PATH=%s' % os.environ['PATH'])

name = 'hypernets-jupyter-widget'
pkg_name = 'hn_widget'

LONG_DESCRIPTION = 'Jupyter widgets for hyperntes'

# Get hn_widget version
version = get_version(pjoin(pkg_name, '_version.py'))

js_dir = pjoin(here, 'js')

# Representative files that should exist after a successful build
jstargets = [
    pjoin(js_dir, 'dist', 'index.js'),
]

data_files_spec = [
    ('share/jupyter/nbextensions/hn_widget', 'hn_widget/nbextension', '*.*'),
    ('share/jupyter/labextensions/hn_widget', 'hn_widget/labextension', '**'),
    ('share/jupyter/labextensions/hn_widget', '.', 'install.json'),
    ('etc/jupyter/nbconfig/notebook.d', '.', 'hn_widget.json'),
]

cmdclass = create_cmdclass('jsdeps', data_files_spec=data_files_spec)
cmdclass['jsdeps'] = combine_commands(
    install_npm(js_dir, npm=['yarn'], build_cmd='build:prod'), ensure_targets(jstargets),
)

setup_args = dict(
    name=name,
    version=version,
    description='Jupyter widgets for hyperntes',
    long_description=LONG_DESCRIPTION,
    include_package_data=True,
    install_requires=[
        'ipywidgets>=7.6.0',
        'numpy',
        'hypernets',
        'pandas'
    ],
    packages=find_packages(),
    zip_safe=False,
    cmdclass=cmdclass,
    author='wuhf',
    author_email='wuhf@zetyun.com',
    url='https://github.com/DataCanvas/hn_widget',
    keywords=[
        'ipython',
        'jupyter',
        'widgets',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: IPython',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

setup(**setup_args)

# -*- coding:utf-8 -*-

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup

version = '0.1.5'

requirements = [
    'numpy>=1.17.4',
    'pandas>=0.25.3',
    'scikit-learn>=0.22.1',
    'lightgbm>=2.2.0'
]

MIN_PYTHON_VERSION = '>=3.6.*'

long_description = open('README.md', encoding='utf-8').read()

extras_require = {
    'dask': ['dask', 'distributed'],
    'cluster': ['paramiko', 'grpcio>=1.24.0', 'protobuf'],
    'tests': ['pytest', ],
}
extras_require["all"] = sorted({v for req in extras_require.values() for v in req})

setup(
    name='hypernets',
    version=version,
    description='An General Automated Machine Learning Framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='DataCanvas Community',
    author_email='yangjian@zetyun.com',
    license='Apache License 2.0',
    install_requires=requirements,
    python_requires=MIN_PYTHON_VERSION,
    extras_require=extras_require,
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests*')),
    package_data={
    },
    zip_safe=False,
    include_package_data=True,
)

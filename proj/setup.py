#!/usr/bin/env python3

from distutils.core import setup

setup(
    name="CgolAI",
    version='0.1',
    description="AI for Conway's Game of Life",
    author='Lucas Wilson',
    author_email='lkwilson96@gmail.com',
    url='https://github.com/larkwt96',
    install_requires=[
        'numpy',
    ],
    packages=['cgolai'],
    package_dir={'cgolai': 'src/cgolai'}
    package_data={'cgolai': os.path.join('res', '*')}
)


#!/usr/bin/env python3

import os
from distutils.core import setup

setup(
    name="cgolai",
    version='0.9',
    description="AI for Conway's Game of Life",
    author='Lucas Wilson',
    author_email='lkwilson96@gmail.com',
    url='https://github.com/larkwt96',
    install_requires=['numpy', 'pygame', 'torch', 'torchvision'],
    packages=['cgolai', 'cgolai.cgol', 'cgolai.ai'],
    package_dir={'cgolai': 'src/cgolai'},
    include_package_data=True,
)

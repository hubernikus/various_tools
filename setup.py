#!/usr/bin/env python
import setuptools
from setuptools import setup

setup(name='vartools',
      version='0.1',
      description='Various Tools',
      author='Lukas Huber',
      author_email='lukas.huber@epfl.ch',
      packages=setuptools.find_packages(where="src", exclude=("tests",)),
      scripts=[],
      package_dir={'': 'src'}
     )

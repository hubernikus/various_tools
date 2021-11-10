#!/usr/bin/env python
from setuptools import setup

setup(name='vartools',
      version='0.1',
      description='Various Tools',
      author='Lukas Huber',
      author_email='lukas.huber@epfl.ch',
      packages=[
          'vartools',
          ],
      scripts=[
          'scripts/main.py',
          ],
      package_dir={'': 'src'}
     )

#!/usr/bin/env python3
# coding=utf-8
from setuptools import setup
from setuptools import find_packages
import os
import shutil

project, version = 'annc', '0.0.1'


def collect_install_patches():
    install_path = os.path.abspath(os.path.join('..', 'install'))
    if os.path.exists('./scripts/patches'):
        shutil.rmtree('./scripts/patches')
    shutil.copytree(install_path, './scripts/patches/')


collect_install_patches()

setup(name=project,
      include_package_data=True,
      version=version,
      description='Accelerated Neural Network Compiler',
      url='https://gitee.com/openeuler/ANNC',
      packages=find_packages() + [
          f'{project}',
          f'{project}/optimize',
          'scripts',
          'scripts.patches',
      ],
      package_data={
          f'scripts.patches': ['*.patch'],
      },
      entry_points={
          'console_scripts': [
              f'{project}-opt = {project}.main:opt',
              f'{project}-apply-tf = scripts.install:tf_install',
          ],
      })

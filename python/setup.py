#!/usr/bin/env python3
# coding=utf-8
import os
import shutil
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

project, version = 'annc', '0.0.3'
BASE_DIR = Path(__file__).resolve().parent


def _collect_install_patches(build_lib: Path):
    install_path = BASE_DIR.parent / 'install'
    if not install_path.exists():
        return
    patches_dst = build_lib / 'scripts' / 'patches'
    if patches_dst.exists():
        shutil.rmtree(patches_dst)
    shutil.copytree(install_path, patches_dst)


def _collect_ops_files(build_lib: Path):
    ops_src = BASE_DIR / 'tensorflow' / 'kernels'
    if not ops_src.exists():
        return
    ops_dst = build_lib / project / 'ops'
    if ops_dst.exists():
        shutil.rmtree(ops_dst)
    shutil.copytree(ops_src, ops_dst)


class build_py(_build_py):
    def run(self):
        super().run()
        build_lib = Path(self.build_lib)
        _collect_install_patches(build_lib)
        _collect_ops_files(build_lib)


setup(
    name=project,
    version=version,
    description='Accelerated Neural Network Compiler',
    url='https://gitee.com/openeuler/ANNC',
    packages=find_packages(exclude=['tests', 'tests.*'])
              + [p for p in (project, f'{project}.optimize', 'scripts')
                 if (BASE_DIR / p.replace('.', '/')).exists()]
              + ['scripts.patches'],
    package_data={
        'scripts.patches': ['*.patch', '*/*.patch', '*/*/*.patch', '*/*/*/*.patch'],
        'annc': ['ops/*.cc'],
    },
    install_requires=[
        'numpy',
        'tensorflow>=2.15',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            f'{project}-opt = {project}.main:opt',
            f'{project}-tf-apply = {project}.tf_apply:main',
        ],
    },
    cmdclass={'build_py': build_py},
)

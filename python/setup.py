# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

def iter_package_data(dir_path):
    packages.append(dir_path.replace('/', '.'))
    package_data[packages[-1]] = ['*']
    for item in os.listdir(dir_path):
        item = os.path.join(dir_path, item).replace('\\', '/')
        if os.path.isdir(item):
            iter_package_data(item)

packages = find_packages()
package_data = {}
iter_package_data('kpgemm/_mlir_libs')
iter_package_data('kpgemm/dialects')
setup(
    name='kpgemm',
    version="1.0.0",
    description='KPGEMM Compiler',
    author='Hou Defu',
    author_email='lark_fluvy@hotmail.com',
    packages=packages,
    install_requires=[],
    include_package_data=True,
    package_data=package_data,
    python_requires='>=3.8.0',
)

# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='flux',
    version='0.1.0',
    description='Data utilities for ML all in one place',
    long_description=readme,
    author='David Chan',
    author_email='davidchan@berkeley.edu',
    url='https://github.com/davidmchan/flux',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
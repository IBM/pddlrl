# -*- coding: utf-8 -*-

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

readme = ''

setup(
    long_description=readme,
    name='pddlrl',
    version='0.1.0',
    python_requires='==3.*',
    author='Clement Gehring',
    author_email='clement@gehring.io',
    license='MIT',
    packages=find_packages(),
    package_data={"pddlrl.experiments": ["*.gin"]},
    install_requires=[
        'chex',
        'dm-acme',
        'dm-env',
        'dm-haiku',
        'gin-config',
        'gym',
        'jax',
        'matplotlib',
        'nlmax',
        'numpy',
        'optax',
        'pandas',
        'pddlenv',
        'pyperplan',
        'ray[tune]',
        'rlax',
        'seaborn',
        'tqdm',
        'ansicolors',
    ],
    extras_require={
        "dev": [
            "mypy",
            "pytest",
            "pytest-timeout",
        ]
    },
)

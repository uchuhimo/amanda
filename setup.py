# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path

readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.rst')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')

setup(
    long_description=readme,
    name='amanda',
    version='0.1.0',
    description='graph instrumentation',
    python_requires='<4,>=3.6',
    project_urls={
        "homepage": "https://github.com/uchuhimo/mmx",
        "repository": "https://github.com/uchuhimo/mmx"
    },
    author='uchuhimo',
    author_email='uchuhimo@outlook.com',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English', 'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    entry_points={"console_scripts": ["amanda = amanda.cli:cli"]},
    packages=[
        'amanda', 'amanda.conversion', 'amanda.core', 'amanda.tests',
        'amanda.tools', 'amanda.tools.byteps', 'amanda.tools.byteps.tensorflow',
        'amanda.tools.debugging', 'amanda.tools.path'
    ],
    package_dir={"": "src"},
    package_data={
        "amanda.tests": ["*.pbtxt"],
        "amanda.tools.debugging": ["*.json", "*.yaml"]
    },
    install_requires=[
        'click', 'contextvars; python_version < "3.7"',
        'dataclasses; python_version < "3.7"', 'immutables', 'mmdnn',
        'more-itertools', 'onnx', 'onnxruntime', 'protobuf', 'scipy', 'six',
        'typing; python_version < "3.7"', 'typing-extensions'
    ],
    extras_require={
        "all": ["tensorflow==1.13.1", "torch==1.4.0", "torchvision==0.5.0"],
        "dev": [
            "black", "bump2version", "coverage[toml]", "dephell[full]",
            "filelock", "fissix", "flake8", "ipython", "isort[pyproject]",
            "jsondiff", "mypy", "pip", "pre-commit", "pytest", "pytest-xdist",
            "sphinx", "tox", "twine", "watchdog", "wget", "wheel"
        ],
        "pytorch": ["torch==1.4.0", "torchvision==0.5.0"],
        "tensorflow": ["tensorflow==1.13.1"]
    },
)

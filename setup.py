#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import re
import subprocess
from distutils.core import Command
from glob import glob
from os.path import basename, splitext
from typing import List

from setuptools import find_packages, setup


# from https://github.com/fastai/fastai/blob/master/setup.py
class DepsCommand(Command):
    """A custom distutils command to print selective dependency groups.
    # show available dependency groups:
    python setup.py -q deps
    # print dependency list for specified groups
    python setup.py -q deps --dep-groups=core,vision
    # see all options:
    python setup.py -q deps --help
    """

    description = "show dependency groups and their packages"
    user_options = [
        # format: (long option, short option, description).
        ("dep-groups=", None, "comma separated dependency groups"),
        ("dep-quote", None, "quote each dependency"),
        ("dep-conda", None, "adjust output for conda"),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.dep_groups = ""
        self.dep_quote = False
        self.dep_conda = False

    def finalize_options(self):
        """Post-process options."""

    def parse(self):
        arg = self.dep_groups.strip()
        return re.split(r" *, *", arg) if len(arg) else []

    def run(self):
        """Run command."""
        wanted_groups = self.parse()

        deps = []
        invalid_groups = []
        for grp in wanted_groups:
            if grp in dep_groups:
                deps.extend(dep_groups[grp])
            else:
                invalid_groups.append(grp)

        if invalid_groups or not wanted_groups:
            print("Available dependency groups:", ", ".join(sorted(dep_groups.keys())))
            if invalid_groups:
                print(f"Error: Invalid group name(s): {', '.join(invalid_groups)}")
                exit(1)
        else:
            # prepare for shell word splitting (no whitespace in items)
            deps = [re.sub(" ", "", x, 0) for x in sorted(set(deps))]
            if self.dep_conda:
                for i in range(len(deps)):
                    # strip pip-specific syntax
                    deps[i] = re.sub(r";.*", "", deps[i])
                    # rename mismatching package names
                    deps[i] = re.sub(r"^torch>", "pytorch>", deps[i])
            if self.dep_quote:
                # for manual copy-n-paste (assuming no " in vars)
                print(" ".join(map(lambda x: f'"{x}"', deps)))
            else:
                # if fed directly to `pip install` via backticks/$() don't quote
                print(" ".join(deps))


class FixCommand(Command):
    description = "fix using pre-commit"
    user_options: List = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(["pre-commit", "run", "-a"])


# note: version is maintained inside src/mmx/version.py
exec(open("src/mmx/version.py").read())
version = __version__  # type: ignore # noqa


# helper functions to make it easier to list dependencies not as a python list,
# but vertically w/ optional built-in comments to
# why a certain version of the dependency is listed
def cleanup(x):
    return re.sub(r" *#.*", "", x.strip())  # comments


def to_list(buffer):
    return list(filter(None, map(cleanup, buffer.splitlines())))


dep_groups = {
    "core": to_list(
        """
    six
    more-itertools
    click
    typing-extensions
    protobuf
    mmdnn
    dataclasses ; python_version<'3.7'
    typing ; python_version<'3.7'
    contextvars ; python_version<'3.7'
    """
    )
}

requirements = [y for x in dep_groups.values() for y in x]

# Environment-specific dependencies.
extras_requirements = {
    "tensorflow": to_list(
        """
    tensorflow
    """
    ),
    "pytorch": to_list(
        """
    torch
    torchvision
    """
    ),
    "dev": to_list(
        """
    flake8
    black
    isort
    mypy
    pre-commit
    pytest
    ipython
    """
    ),
}

# need at least setuptools>=36.2 to support syntax:
#   dataclasses ; python_version<'3.7'
setup_requirements = to_list(
    """
    pytest-runner
    setuptools>=36.2
"""
)

test_requirements = to_list(
    """
    pytest
"""
)

# Meta dependency groups.
all_deps: List[str] = []
for group_name in extras_requirements:
    all_deps += extras_requirements[group_name]
extras_requirements["all"] = all_deps

setup(
    cmdclass={"deps": DepsCommand, "fix": FixCommand},
    name="mmx",
    version=version,
    license="Apache Software License 2.0",
    description="graph instrumentation",
    author="uchuhimo",
    author_email="uchuhimo@outlook.com",
    url="https://github.com/uchuhimo/mmx",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        # uncomment if you test on these interpreters:
        # "Programming Language :: Python :: Implementation :: PyPy",
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        "Topic :: Utilities",
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    setup_requires=setup_requirements,
    install_requires=requirements,
    extras_require=extras_requirements,
    tests_require=test_requirements,
    python_requires=">=3.6",
    entry_points={"console_scripts": ["mmx = mmx.cli:mmx"]},
)

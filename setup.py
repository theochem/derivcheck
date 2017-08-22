#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Derivcheck is robust and very sensitive tester for analytic derivatives.
# Copyright (C) 2017 Toon Verstraelen <Toon.Verstraelen@UGent.be>.
#
# This file is part of Derivcheck.
#
# Derivcheck is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Derivcheck is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Derivcheck setup script.

If you are not familiar with setup.py, just use pip instead:

    pip install derivcheck --user --upgrade

Alternatively, you can install from source with

    ./setup.py install --user
"""

from __future__ import print_function

from setuptools import setup

from tools.gitversion import get_gitversion


def readme():
    """Load README.rst for display on PyPI."""
    with open('README.rst') as f:
        return f.read()

setup(
    name='derivcheck',
    version=get_gitversion('derivcheck', verbose=(__name__ == '__main__')),
    description='A robust and very sensitive tester for analytic derivatives.',
    long_description=readme(),
    author='Toon Verstraelen',
    author_email='Toon.Verstraelen@UGent.be',
    url='https://github.com/theochem/derivcheck',
    packages=['derivcheck'],
    install_requires=['numpy', 'nose'],
    python_requires='>=2.7',
    zip_safe=False,
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
)

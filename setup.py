#!/usr/bin/env python
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
#
# --
"""Derivcheck install script.

If you are not familiar with setup.py, just use pip instead:

    pip install derivcheck --user --upgrade

Alternatively, you can install from source with

    ./setup.py install --user
"""

from setuptools import setup
from derivcheck import __version__

setup(
    name='derivcheck',
    version=__version__,
    description='A robust and very sensitive tester for analytic derivatives.',
    author='Toon Verstraelen',
    author_email='Toon.Verstraelen@UGent.be',
    url='https://github.com/tovrstra/derivcheck',
    py_modules=['derivcheck'],
    install_requires=['numpy'],
    python_requires='==2.7.*',
    setup_requires=['nose>=1.0', 'coverage'],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
    ],
)

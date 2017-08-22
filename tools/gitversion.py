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

import os
import sys
import subprocess


__all__ = ['get_gitversion']


VERSION_TEMPLATE = """\
\"""Do not edit this file, versioning is governed by ``git describe --tags`` and ``setup.py``.\"""
__version__ = '{}'
"""


def get_gitversion(pypkg, verbose):
    # Try to get the version from git describe
    version = None
    try:
        if verbose:
            print('Trying to get the version from git describe')
        git_describe = subprocess.check_output(["git", "describe", "--tags"])
        version_words = git_describe.decode('utf-8').strip().split('-')
        version = version_words[0]
        if len(version_words) > 1:
            version += '.post' + version_words[1]
        if verbose:
            print('Version from git describe: {}'.format(version))
    except (subprocess.CalledProcessError, OSError):
        pass

    # Interact with version.py
    fn_version = os.path.join(os.path.dirname(__file__), '..', pypkg, 'version.py')
    if version is None:
        if verbose:
            print('Trying to get the version from {}',format(fn_version))
        # Try to load the git version tag from version.py
        try:
            with open(fn_version, 'r') as fh:
                version = fh.read().split('=')[-1].replace('\'', '').strip()
        except IOError:
            print('Could not determine version. Giving up.')
            sys.exit(1)
        if verbose:
            print('Version according to {}: {}'.format(fn_version, version))
    else:
        # Store the git version tag in version.py
        if verbose:
            print('Writing version to {}'.format(fn_version))
        with open(fn_version, 'w') as fh:
            fh.write(VERSION_TEMPLATE.format(version))

    return version

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
"""Very basic usage example of Derivcheck."""


import numpy as np
from derivcheck import derivcheck


def function(x):
    """Compute a trivial quadratic function."""
    return 0.5*(x**2).sum()


def gradient(x):
    """Compute analytic partial derivatives of ``function``."""
    return x


def main():
    """Run the example."""
    # Some reference points at which the derivative must be tested.
    xs = [np.random.normal(0, 1, (4, 3)) for i_ in xrange(10)]

    # Test the derivatives at the reference points. See docstring of derivcheck for optional
    # arguments.
    derivcheck(function, gradient, xs)


if __name__ == '__main__':
    main()

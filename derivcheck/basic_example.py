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
# --
"""Very basic usage example of Derivcheck."""


import numpy as np

from derivcheck import assert_deriv


def function(arg):
    """Compute a trivial quadratic function."""
    return 0.5*(arg**2).sum()


def gradient(arg):
    """Compute analytic partial derivatives of ``function``."""
    return arg


def main():
    """Run the example."""
    # Some reference pointsat which the derivative must be tested.
    origin = np.random.normal(0, 1, (4, 3))

    # Test the derivatives at the reference points. See docstring for optional arguments.
    assert_deriv(function, gradient, origin)


if __name__ == '__main__':  # pragma: no cover
    main()

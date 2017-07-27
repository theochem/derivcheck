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


from contextlib import contextmanager
from exceptions import FloatingPointError

from nose.tools import assert_raises
import numpy as np

from derivcheck import derivcheck, _random_unit
from basic_example import main as example_main


def check_derivcheck_0d(nx, x_shape):
    f = lambda x: 0.5 * np.sum(x**2)
    g = lambda x: x
    xs = [np.random.normal(0, 1, x_shape) for ix in xrange(nx)]
    derivcheck(f, g, xs, verbose=True)


def test_derivcheck_0d():
    yield check_derivcheck_0d, 1, None
    yield check_derivcheck_0d, 1, (10, )
    yield check_derivcheck_0d, 1, (3, 4)
    yield check_derivcheck_0d, 10, None
    yield check_derivcheck_0d, 10, (10, )
    yield check_derivcheck_0d, 10, (3, 4)


def check_derivcheck_nd(nx, x_shape):
    f = lambda x: 0.5 * x**2

    def g(x):
        result = np.zeros(x_shape + x_shape)
        for idx, val in np.lib.index_tricks.ndenumerate(x):
            result[idx + idx] = val
        return result

    xs = [np.random.normal(0, 1, x_shape) for ix in xrange(nx)]
    derivcheck(f, g, xs, verbose=True)


def test_derivcheck_nd():
    yield check_derivcheck_nd, 1, (10, )
    yield check_derivcheck_nd, 1, (3, 4)
    yield check_derivcheck_nd, 10, (10, )
    yield check_derivcheck_nd, 10, (3, 4)


def check_derivcheck_extra1(nx):
    f = lambda x: 0.5 * (x**2).sum(axis=1)

    def g(x):
        result = np.zeros((4, 4, 3), float)
        for i0 in xrange(4):
            for i1 in xrange(3):
                result[i0, i0, i1] = x[i0, i1]
        return result

    xs = [np.random.normal(0, 1, (4, 3)) for ix in xrange(nx)]
    derivcheck(f, g, xs, verbose=True)


def test_derivcheck_extra1():
    yield check_derivcheck_extra1, 1
    yield check_derivcheck_extra1, 10


def check_derivcheck_nd_zeros(nx, x_shape):
    f = lambda x: np.ones(x_shape)
    g = lambda x: np.zeros(x_shape + x_shape)
    xs = [np.random.normal(0, 1, x_shape) for ix in xrange(nx)]
    derivcheck(f, g, xs, verbose=True)


def test_derivcheck_nd_zeros():
    yield check_derivcheck_nd_zeros, 1, (10, )
    yield check_derivcheck_nd_zeros, 1, (3, 4)
    yield check_derivcheck_nd_zeros, 10, (10, )
    yield check_derivcheck_nd_zeros, 10, (3, 4)


def test_derivcheck_nd_weights():

    # function is indeterminate for x[0] <= 1
    def f(x):
        with np.errstate(divide='raise'):
            return x[1] / max(0, x[0] - 1) + x[2]

    # gradient is indeterminate for x[0] <= 1
    def g(x):
        with np.errstate(divide='raise'):
            return np.array([-x[1] / ((x[0] - 1)**2), 1 / max(0, x[0] - 1), 1.0])

    # do searches near the indeterminate region
    xs = np.array([1.03, 4.0, 1.0])
    # romin searches into x[0] < 1
    assert_raises(FloatingPointError, derivcheck, f, g, xs, 0.1, 16)
    # reduce weight on x[0] so that romin does not search so far
    weights = np.array([1.e-4, 1.0, 1.0])
    derivcheck(f, g, xs, 0.1, 16, weights=weights, verbose=True)


def test_example():
    example_main()


def test_derivcheck_exceptions():
    with assert_raises(ValueError):
        derivcheck(None, None, [0.0], 0.1, 7)
    with assert_raises(NotImplementedError):
        derivcheck(None, None, [None], 0.1, 7)


def test_random_unit():
    weights = np.array([[3, -1], [0.0, 2]])
    np.testing.assert_almost_equal(np.linalg.norm(_random_unit(weights.shape, weights)), 1)

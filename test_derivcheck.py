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
"""Unit tests for derivcheck."""


from builtins import range

from nose.tools import assert_raises
import numpy as np

from derivcheck import derivcheck, _random_unit
from basic_example import main as example_main


def _check_derivcheck_0d(narg, x_shape):
    _function = lambda arg: 0.5 * np.sum(arg**2)
    _gradient = lambda arg: arg
    args = [np.random.normal(0, 1, x_shape) for _ in range(narg)]
    derivcheck(_function, _gradient, args, verbose=True)


def test_derivcheck_0d():
    yield _check_derivcheck_0d, 1, None
    yield _check_derivcheck_0d, 1, (10, )
    yield _check_derivcheck_0d, 1, (3, 4)
    yield _check_derivcheck_0d, 10, None
    yield _check_derivcheck_0d, 10, (10, )
    yield _check_derivcheck_0d, 10, (3, 4)


def _check_derivcheck_nd(narg, x_shape):
    _function = lambda arg: 0.5 * arg**2

    def _gradient(arg):
        result = np.zeros(x_shape + x_shape)
        for idx, val in np.lib.index_tricks.ndenumerate(arg):
            result[idx + idx] = val
        return result

    args = [np.random.normal(0, 1, x_shape) for _ in range(narg)]
    derivcheck(_function, _gradient, args, verbose=True)


def test_derivcheck_nd():
    yield _check_derivcheck_nd, 1, (10, )
    yield _check_derivcheck_nd, 1, (3, 4)
    yield _check_derivcheck_nd, 10, (10, )
    yield _check_derivcheck_nd, 10, (3, 4)


def _check_derivcheck_extra1(narg):
    _function = lambda arg: 0.5 * (arg**2).sum(axis=1)

    def _gradient(arg):
        result = np.zeros((4, 4, 3), float)
        for index0 in range(4):
            for index1 in range(3):
                result[index0, index0, index1] = arg[index0, index1]
        return result

    args = [np.random.normal(0, 1, (4, 3)) for _ in range(narg)]
    derivcheck(_function, _gradient, args, verbose=True)


def test_derivcheck_extra1():
    yield _check_derivcheck_extra1, 1
    yield _check_derivcheck_extra1, 10


def _check_derivcheck_nd_zeros(narg, x_shape):
    function = lambda arg: np.ones(x_shape)
    gradient = lambda arg: np.zeros(x_shape + x_shape)
    args = [np.random.normal(0, 1, x_shape) for _ in range(narg)]
    derivcheck(function, gradient, args, verbose=True)


def test_derivcheck_nd_zeros():
    yield _check_derivcheck_nd_zeros, 1, (10, )
    yield _check_derivcheck_nd_zeros, 1, (3, 4)
    yield _check_derivcheck_nd_zeros, 10, (10, )
    yield _check_derivcheck_nd_zeros, 10, (3, 4)


def test_derivcheck_nd_weights():

    # function is indeterminate for arg[0] <= 1
    def _function(arg):
        with np.errstate(divide='raise'):
            return arg[1] / max(0, arg[0] - 1) + arg[2]

    # gradient is indeterminate for arg[0] <= 1
    def _gradient(arg):
        with np.errstate(divide='raise'):
            return np.array([-arg[1] / (arg[0] - 1)**2, 1 / max(0, arg[0] - 1), 1.0])

    # do searches near the indeterminate region
    args = np.array([1.03, 4.0, 1.0])
    # romin searches into arg[0] < 1
    with assert_raises(FloatingPointError):
        derivcheck(_function, _gradient, args, 0.1, 16)
    # reduce weight on arg[0] so that romin does not search so far
    weights = np.array([1.e-4, 1.0, 1.0])
    derivcheck(_function, _gradient, args, 0.1, 16, weights=weights, verbose=True)


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

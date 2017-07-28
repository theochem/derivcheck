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


from builtins import range  # pylint: disable=redefined-builtin

from nose.tools import assert_raises
import numpy as np

# We want to test the wildcard import!
from derivcheck import *  # pylint: disable=wildcard-import
from basic_example import main as example_main


def test_ridders_corner_cases():
    with assert_raises(ValueError):
        diff_ridders(np.sin, 0.0, 0.0)
    with assert_raises(ValueError):
        diff_ridders(np.sin, 0.0, 1.0, con=0.9)
    with assert_raises(ValueError):
        diff_ridders(np.sin, 0.0, 1.0, safe=0.9)
    assert diff_ridders(np.sin, 0.0, 1.0, maxiter=0) == (None, None)


def test_ridders_simple():
    for arg in np.linspace(-1.0, 1.0, 15):
        deriv, error = diff_ridders(np.exp, arg, 0.1)
        assert error < 1e-10
        np.testing.assert_allclose(deriv, np.exp(arg))
        deriv, error = diff_ridders(np.sin, arg, 0.1)
        assert error < 1e-10
        np.testing.assert_allclose(deriv, np.cos(arg))


def _check_assert_deriv_0d_harm(nrep, arg_shape):
    _function = lambda arg: 0.5 * np.sum(arg**2)
    _gradient = lambda arg: arg
    for _ in range(nrep):
        origin = np.random.normal(0, 1, arg_shape)
        assert_deriv(_function, _gradient, origin)


def test_assert_deriv_0d_harm():
    yield _check_assert_deriv_0d_harm, 10, ()
    yield _check_assert_deriv_0d_harm, 10, (10, )
    yield _check_assert_deriv_0d_harm, 10, (3, 4)
    yield _check_assert_deriv_0d_harm, 10, (2, 3, 4)


def _check_assert_deriv_0d_exp(nrep, arg_shape):
    _function = lambda arg: np.exp(arg).sum()
    for _ in range(nrep):
        origin = np.random.uniform(-1.0, 1.0, arg_shape)
        assert_deriv(_function, np.exp, origin)


def test_assert_deriv_0d_exp():
    yield _check_assert_deriv_0d_exp, 10, ()
    yield _check_assert_deriv_0d_exp, 10, (10, )
    yield _check_assert_deriv_0d_exp, 10, (3, 4)
    yield _check_assert_deriv_0d_exp, 10, (2, 3, 4)


def _check_assert_deriv_nd(nrep, arg_shape, output_mask):
    _function = lambda arg: 0.5 * arg**2

    def _gradient(arg):
        result = np.zeros(arg_shape + arg_shape)
        for idx, val in np.lib.index_tricks.ndenumerate(arg):
            result[idx + idx] = val
        return result

    for _ in range(nrep):
        origin = np.random.normal(0, 1, arg_shape)
        assert_deriv(_function, _gradient, origin, output_mask=output_mask)


def test_assert_deriv_nd():
    yield _check_assert_deriv_nd, 10, (), None
    yield _check_assert_deriv_nd, 10, (10, ), None
    yield _check_assert_deriv_nd, 10, (3, 4), None
    yield _check_assert_deriv_nd, 10, (2, 3, 4), None
    yield _check_assert_deriv_nd, 10, (5, ), np.array([1, 1, 0, 0, 1], dtype=bool)
    yield _check_assert_deriv_nd, 10, (2, 2), np.array([[1, 0], [0, 1]], dtype=bool)


def _check_assert_deriv_extra1(nrep, arg_shape, output_mask):
    _function = lambda arg: 0.5 * (arg**2).sum(axis=-1)

    def _gradient(arg):
        result = np.zeros(arg_shape[:-1] + arg_shape, float)
        for idx, val in np.lib.index_tricks.ndenumerate(arg):
            result[idx[:-1] + idx] = val
        return result

    for _ in range(nrep):
        arg = np.random.normal(0, 1, arg_shape)
        assert_deriv(_function, _gradient, arg, output_mask=output_mask)


def test_assert_deriv_extra1():
    yield _check_assert_deriv_extra1, 10, (3,), None
    yield _check_assert_deriv_extra1, 10, (4, 3), None
    yield _check_assert_deriv_extra1, 10, (2, 4, 3), None
    yield _check_assert_deriv_extra1, 10, (4, 3), np.array([1, 0, 0, 1], dtype=bool)
    yield _check_assert_deriv_extra1, 10, (2, 2, 3), np.array([[1, 1], [0, 0]], dtype=bool)


def _check_assert_deriv_nd_zeros(nrep, arg_shape, output_mask):
    _function = lambda arg: np.ones(arg_shape)
    _gradient = lambda arg: np.zeros(arg_shape + arg_shape)
    for _ in range(nrep):
        args = np.random.normal(0, 1, arg_shape)
        assert_deriv(_function, _gradient, args, output_mask=output_mask)


def test_assert_deriv_nd_zeros():
    yield _check_assert_deriv_nd_zeros, 10, (), None
    yield _check_assert_deriv_nd_zeros, 10, (10, ), None
    yield _check_assert_deriv_nd_zeros, 10, (3, 4), None
    yield _check_assert_deriv_nd_zeros, 10, (2, 3, 4), None
    yield _check_assert_deriv_nd_zeros, 10, (5, ), np.array([0, 1, 1, 0, 0], dtype=bool)
    yield _check_assert_deriv_nd_zeros, 10, (2, 2), np.array([[0, 1], [1, 0]], dtype=bool)


def test_assert_deriv_nd_weights():

    # function is indeterminate for arg[0] <= 1
    def _function(arg):
        with np.errstate(divide='raise'):
            return arg[1] / max(0, arg[0] - 1) + arg[2]

    # gradient is indeterminate for arg[0] <= 1
    def _gradient(arg):
        with np.errstate(divide='raise'):
            return np.array([-arg[1] / (arg[0] - 1)**2, 1 / max(0, arg[0] - 1), 1.0])

    # do searches near the indeterminate region
    arg = np.array([1.03, 4.0, 1.0])
    # romin searches into arg[0] < 1
    with assert_raises(FloatingPointError):
        assert_deriv(_function, _gradient, arg, 0.1)
    # reduce widths on arg[0] so that romin does not search so far
    widths = np.array([1.e-4, 1.0, 1.0])
    assert_deriv(_function, _gradient, arg, widths)
    # zero width on arg[0] so that it gets skipped
    widths = np.array([0.0, 1.0, 1.0])
    assert_deriv(_function, _gradient, arg, widths)


def test_assert_deriv_corner_cases():
    _function = lambda arg: np.exp(arg).sum()
    arg = np.ones((3, 3))
    with assert_raises(FloatingPointError):
        assert_deriv(_function, np.exp, arg, 0.1, rtol=0, atol=0)


def test_example():
    example_main()

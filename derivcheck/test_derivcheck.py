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
"""Unit tests for derivcheck."""

from nose.tools import assert_raises
import numpy as np

from derivcheck import diff_ridders, assert_deriv
from .basic_example import main as example_main


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


def _check_assert_deriv_0d_harm(nrep, arg_shape, numtested):
    _function = lambda arg: 0.5 * np.sum(arg**2)
    _gradient = lambda arg: arg
    for _ in range(nrep):
        origin = np.random.normal(0, 1, arg_shape)
        assert assert_deriv(_function, _gradient, origin) == numtested


def test_assert_deriv_0d_harm():
    yield _check_assert_deriv_0d_harm, 10, (), 1
    yield _check_assert_deriv_0d_harm, 10, (10, ), 10
    yield _check_assert_deriv_0d_harm, 10, (3, 4), 12
    yield _check_assert_deriv_0d_harm, 10, (2, 3, 4), 24


def _check_assert_deriv_0d_exp(nrep, arg_shape, numtested):
    _function = lambda arg: np.exp(arg).sum()
    for _ in range(nrep):
        origin = np.random.uniform(-1.0, 1.0, arg_shape)
        assert assert_deriv(_function, np.exp, origin) == numtested


def test_assert_deriv_0d_exp():
    yield _check_assert_deriv_0d_exp, 10, (), 1
    yield _check_assert_deriv_0d_exp, 10, (10, ), 10
    yield _check_assert_deriv_0d_exp, 10, (3, 4), 12
    yield _check_assert_deriv_0d_exp, 10, (2, 3, 4), 24


def _check_assert_deriv_nd(nrep, arg_shape, output_mask, numtested):
    _function = lambda arg: 0.5 * arg**2

    def _gradient(arg):
        result = np.zeros(arg_shape + arg_shape)
        for idx, val in np.lib.index_tricks.ndenumerate(arg):
            result[idx + idx] = val
        return result

    for _ in range(nrep):
        origin = np.random.normal(0, 1, arg_shape)
        assert assert_deriv(_function, _gradient, origin, output_mask=output_mask) == numtested


def test_assert_deriv_nd():
    yield _check_assert_deriv_nd, 10, (), None, 1
    yield _check_assert_deriv_nd, 10, (10, ), None, 100
    yield _check_assert_deriv_nd, 10, (3, 4), None, 144
    yield _check_assert_deriv_nd, 10, (2, 3, 4), None, 576
    yield _check_assert_deriv_nd, 10, (5, ), np.array([1, 1, 0, 0, 1], dtype=bool), 15
    yield _check_assert_deriv_nd, 10, (2, 2), np.array([[1, 0], [0, 1]], dtype=bool), 8


def _check_assert_deriv_extra1(nrep, arg_shape, output_mask, numtested):
    _function = lambda arg: 0.5 * (arg**2).sum(axis=-1)

    def _gradient(arg):
        result = np.zeros(arg_shape[:-1] + arg_shape, float)
        for idx, val in np.lib.index_tricks.ndenumerate(arg):
            result[idx[:-1] + idx] = val
        return result

    for _ in range(nrep):
        arg = np.random.normal(0, 1, arg_shape)
        assert assert_deriv(_function, _gradient, arg, output_mask=output_mask) == numtested


def test_assert_deriv_extra1():
    yield _check_assert_deriv_extra1, 10, (3,), None, 3
    yield _check_assert_deriv_extra1, 10, (4, 3), None, 48
    yield _check_assert_deriv_extra1, 10, (2, 4, 3), None, 192
    yield _check_assert_deriv_extra1, 10, (4, 3), np.array([1, 0, 0, 1], dtype=bool), 24
    yield _check_assert_deriv_extra1, 10, (2, 2, 3), np.array([[1, 1], [0, 0]], dtype=bool), 24


def _check_assert_deriv_nd_zeros(nrep, arg_shape, output_mask, numtested):
    _function = lambda arg: np.ones(arg_shape)
    _gradient = lambda arg: np.zeros(arg_shape + arg_shape)
    for _ in range(nrep):
        args = np.random.normal(0, 1, arg_shape)
        assert assert_deriv(_function, _gradient, args, output_mask=output_mask) == numtested


def test_assert_deriv_nd_zeros():
    yield _check_assert_deriv_nd_zeros, 10, (), None, 1
    yield _check_assert_deriv_nd_zeros, 10, (10, ), None, 100
    yield _check_assert_deriv_nd_zeros, 10, (3, 4), None, 144
    yield _check_assert_deriv_nd_zeros, 10, (2, 3, 4), None, 576
    yield _check_assert_deriv_nd_zeros, 10, (5, ), np.array([0, 1, 1, 0, 0], dtype=bool), 10
    yield _check_assert_deriv_nd_zeros, 10, (2, 2), np.array([[0, 1], [1, 0]], dtype=bool), 8


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
    assert assert_deriv(_function, _gradient, arg, widths) == 3
    # zero width on arg[0] so that it gets skipped
    widths = np.array([0.0, 1.0, 1.0])
    assert assert_deriv(_function, _gradient, arg, widths) == 2


def test_assert_deriv_corner_cases():
    _function = lambda arg: np.exp(arg).sum()
    arg = np.ones((3, 3))
    with assert_raises(FloatingPointError):
        assert_deriv(_function, np.exp, arg, 0.1, rtol=0, atol=0)


def test_example():
    example_main()

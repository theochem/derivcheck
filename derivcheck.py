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
"""Robust and sensitive tester for first-order analytic partial derivatives."""

from __future__ import print_function

from builtins import range, object  # pylint: disable=redefined-builtin

import numpy as np


__all__ = ['diff_ridder', 'assert_deriv']

__version__ = '1.0.0'


def diff_ridder(function, x, h, con=1.4, safe=2.0, maxiter=15):
    """Estimate first-order derivative with Ridder's finite difference method.

    This implementation is based on the one from the book Numerical Recipes. The code
    is pythonized and no longer using fixed-size arrays. Also, the output of the function
    can be an array.

    Parameters
    ----------
    function : function
        The function to be differentiated.
    x : float
        The point at which must be differentiated.
    h : float
        The initial step size.
    con : float
        The rate at which the step size is decreased (contracted). Must be larger than
        one.
    safe : float
        The safety check used to terminate the algorithm. If Errors between successive
        orders become larger than ``safe`` times the error on the best estimate, the
        algorithm stop. This happens due to round-off errors.
    maxiter : int
        The maximum number of iterations, equals the maximum number of function calls and
        also the highest polynomial order in the Neville method.

    Returns
    -------
    estimate : float
        The best estimate of the first-order derivative.
    error : float
        The (optimistic) estimate of the error on the derivative.

    """
    if h == 0.0:
        raise ValueError('h must be nonzero.')
    if con <= 1.0:
        raise ValueError('con must be larger than one.')
    if safe <= 1.0:
        raise ValueError('safe must be larger than one.')

    con2 = con*con
    table = [[(function(x + h) - function(x - h))/(2.0*h)]]
    estimate = None
    error = None

    # Loop based on Neville's method.
    # Successive rows in the table will go to smaller stepsizes.
    # Successive columns in the table go to higher orders of extrapolation.
    for i in range(1, maxiter):
        # Reduce step size.
        h /= con;
        # First-order approximation at current step size.
        table.append([(
            np.asarray(function(x + h)) - np.asarray(function(x - h))
        )/(2.0*h)])
        # Compute higher-orders
        fac = con2
        for j in range(1, i+1):
            # Compute extrapolations of various orders, requiring no new
            # function evaluations. This is a recursion relation based on
            # Neville's method.
            table[i].append((table[i][j-1]*fac - table[i-1][j-1])/(fac-1.0));
            fac = con2*fac;

            # The error strategy is to compare each new extrapolation to one
            # order lower, both at the present stepsize and the previous one:
            current_error = max(abs(table[i][j] - table[i][j-1]).max(),
                                abs(table[i][j] - table[i-1][j-1]).max())

            # If error has decreased, save the improved estimate.
            if error is None or current_error <= error:
                error = current_error
                estimate = table[i][j]

        # If the highest-order estimate is growing larger than the error on the best
        # estimate, the algorithm becomes numerically instable. Time to quit.
        if abs(table[i][i] - table[i-1][i-1]).max() >= safe * error:
            break
        i += 1
    return estimate, error


class OneDimWrapper(object):
    def __init__(self, function, origin, axis):
        self.function = function
        self.origin = origin
        self.axis = axis

    def __call__(self, x):
        return self.function(self.origin + self.axis*x)


def assert_deriv(function, gradient, origin, widths=1e-4, output_mask=None, rtol=1e-5, atol=1e-8):
    """Test the gradient of a function.

    Parameters
    ----------
    function : function
        The function whose derivatives must be tested.
    gradient : function
        Computes the gradient of the function, to be tested.
    origin : np.ndarray
        The point at which the derivatives are computed.
    widths : float or np.ndarray
        The initial (maximal) step size for the finite difference method. Do not take
        a value that is too small. When an array, each matrix element of the input of the
        function gets a different step size. Set to zero to skip an element.
    output_mask : np.ndarray or None
        When given, selects the outputs of function to be tested. Only relevant for
        functions that return arrays.
    rtol : float
        The allowed relative error on the derivative.
    atol : float
        The allowed absolute error on the derivative.

    Raises
    ------
    AssertionError when the error on the derivative is too large.

    """
    origin = np.asarray(origin)
    gradient = np.asarray(gradient(origin))
    if output_mask is not None:
        gradient = gradient[output_mask]
    else:
        gradient = gradient.reshape(-1, origin.size)

    for iaxis in range(origin.size):
        if origin.ndim == 0:
            indices = ()
        else:
            indices = np.unravel_index(iaxis, origin.shape)
        axis = np.zeros(origin.shape)
        axis[indices] = 1.0
        if isinstance(widths, float):
            h = widths
        else:
            h = widths[indices]
        if h > 0:
            wrapper = OneDimWrapper(function, origin, axis)
            deriv_approx, deriv_error = diff_ridder(wrapper, 0.0, h)
            if output_mask is None:
                deriv_approx = deriv_approx.ravel()
            else:
                deriv_approx = deriv_approx[output_mask]
            deriv = gradient[:, iaxis]
            if deriv_error >= atol and deriv_error >= rtol*abs(deriv).max():
                raise AssertionError('Inaccurate estimate of derivative.')
            print(iaxis, deriv, deriv_approx)
            err_msg = 'case {} {}'.format(iaxis, indices)
            np.testing.assert_allclose(deriv, deriv_approx, rtol, atol, err_msg=err_msg)

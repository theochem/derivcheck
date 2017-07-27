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


import numpy as np


__all__ = ["derivcheck"]


# Gauss-Legendre quadrature grids (points and weights) for different orders.
gauss_legendre = {
    2: (np.array([-5.773502691896258e-01, 5.773502691896257e-01]),
        np.array([1.000000000000000e+00, 1.000000000000000e+00])),
    4: (np.array([-8.611363115940527e-01, -3.399810435848563e-01, 3.399810435848563e-01,
                  8.611363115940526e-01]),
        np.array([3.478548451374538e-01, 6.521451548625461e-01, 6.521451548625461e-01,
                  3.478548451374538e-01])),
    8: (np.array([-9.602898564975363e-01, -7.966664774136268e-01, -5.255324099163290e-01,
                  -1.834346424956498e-01, 1.834346424956498e-01, 5.255324099163289e-01,
                  7.966664774136267e-01, 9.602898564975362e-01]),
        np.array([1.012285362903763e-01, 2.223810344533745e-01, 3.137066458778873e-01,
                  3.626837833783619e-01, 3.626837833783619e-01, 3.137066458778873e-01,
                  2.223810344533745e-01, 1.012285362903763e-01])),
    16: (np.array([-9.894009349916499e-01, -9.445750230732326e-01, -8.656312023878318e-01,
                   -7.554044083550031e-01, -6.178762444026438e-01, -4.580167776572274e-01,
                   -2.816035507792589e-01, -9.501250983763744e-02, 9.501250983763743e-02,
                   2.816035507792589e-01, 4.580167776572274e-01, 6.178762444026437e-01,
                   7.554044083550030e-01, 8.656312023878316e-01, 9.445750230732325e-01,
                   9.894009349916498e-01]),
         np.array([2.715245941175409e-02, 6.225352393864789e-02, 9.515851168249277e-02,
                   1.246289712555339e-01, 1.495959888165767e-01, 1.691565193950025e-01,
                   1.826034150449236e-01, 1.894506104550685e-01, 1.894506104550685e-01,
                   1.826034150449236e-01, 1.691565193950025e-01, 1.495959888165767e-01,
                   1.246289712555339e-01, 9.515851168249277e-02, 6.225352393864789e-02,
                   2.715245941175409e-02])),
}


def _deriv_error(f, g, x, eps_x=1e-4, order=8):
    """Compute the error between difference of f at two points and its FD approximation.

    Parameters
    ----------
    f : function
        Computes the function value for a given x.
    g : function
        Computes the derivative for a given x.
    x : float
        The center of the interval at which the test is performed.
    eps_x : float
        The half width of the interval.
    order : int (2, 4, 8, 16)
        The number of grid points in the quadrature.

    This function computes the difference of f(x+eps_x) - f(x-eps_x). It also computes
    the integral of the derivative with Gaussian quadrature, which should be very close
    to the former.

    The functions f and g may return scalars or arrays. The return values will have
    compatible data types.

    Returns
    -------
    delta : float or np.ndarray
            The difference between f at the end points of the interval.
    delta_approx : float or np.ndarray
                   The approximation of delta computed with the derivative, ``g``.

    """
    # Get the right quadrature points and weights
    if order not in gauss_legendre:
        raise ValueError('The order must be one of %s' % gauss_legendre.keys())
    points, weights = gauss_legendre.get(order)
    # Compute the difference between f at two different points
    delta = f(x + eps_x) - f(x - eps_x)
    # Approximate that difference with Gaussian quadrature, with some sanity checks
    derivs = np.array([g(x + eps_x*p) for p in points])
    assert delta.shape == derivs.shape[1:delta.ndim+1]
    if len(derivs.shape) > 1:
        assert derivs.shape[1:] == delta.shape
    delta_approx = np.tensordot(weights, derivs, axes=1)*eps_x
    # Done
    return delta, delta_approx


def _deriv_error_array(f, g, x, eps_x=1e-4, order=8, nrep=None, weights=1):
    """Extension of deriv_error for functions that take arrays as arguments.

    This function performs many one-dimensional tests with _deriv_error along randomly
    chosen directions.

    Parameters
    ----------
    f : function
        Computes the function value for a given x.
    g : function
        Computes the derivative for a given x.
    x : np.ndarray
        The reference point for multiple calls to _deriv_error.
    eps_x : float
        The half width of the interval for _deriv_error.
    order : int (2, 4, 8, 16)
        The number of grid points in the quadrature.
    nrep : int
        The number of random directions. [default=x.size**2]
    weights : np.ndarray
        An array with the same shape as x, specifies which directions should be scanned
        most often.

    Returns
    -------
    delta : float or np.ndarray
        The difference between f at the end points of the interval, for multiple random
        directions
    delta_approx : float or np.ndarray
        The approximation of delta computed with the derivative, ``g``, for multiple
        random directions.

    """
    if nrep is None:
        nrep = x.size**2
    # run different random line scans
    results = []
    for irep in xrange(nrep):
        # Generate a generalized random unit vector.
        while True:
            unit = np.random.normal(0, 1, x.shape)*weights
            norm = np.sqrt((unit**2).sum())
            if norm > 1e-3:
                unit /= norm
                break

        # Define f and g along the one-dimensional scan
        def f_scan(x_scan):
            return f(x_scan*unit + x)

        def g_scan(x_scan):
            # nasty chain rule
            return np.tensordot(g(x_scan*unit + x), unit, axes=unit.ndim)

        # Collect results
        results.append(_deriv_error(f_scan, g_scan, 0.0, eps_x, order))
    return results


def derivcheck(f, g, xs, eps_x=1e-4, order=8, nrep=None, rel_ftol=1e-3, weights=1,
               discard=0.1, verbose=False):
    """Checker for the implementation of partial derivatives.

    This function performs a Gaussian quadrature using g as integrand to approximate
    differences between function values of f. The interval for the integration is a small
    range around one or more reference points, xs. If the argument of f and g is an array,
    random line scans are done around the reference point.

    Parameters
    ----------
    f : function
        Computes the function value for a given x.
    g : function
        Computes the derivative for a given x.
    xs : float, np.ndarray, or list thereof
        Reference point(s) for _deriv_error or _deriv_error_array. If only one float or
        array is given, it is the only reference point. When a list is given, _deriv_error
        or derive_error_array is called for every reference point in the list.
    eps_x : float
        The half width of the interval for deriv_error or deriv_error_array.
    order : int (2, 4, 8, 16)
        The number of grid points in the quadrature.
    nrep : int
        The number of random directions for one reference point, in case xs is an array.
        It is ignored otherwise. [default=xs.size**2]
    rel_ftol : float
        The allowed relative error between delta and delta_approx. [default=1e-3]
    weights : np.ndarray
        An array with the same shape as xs, specifies which directions should be scanned
        more often. [default=1]
    discard : float
        The fraction of smallest deltas to discard, together with there corresponding
        deltas_approx. [default=0.1]
    verbose : bool
        If True, some info is printed on screen. [default=False].

    Raises
    ------
    AssertionError when any of the selected (delta, delta_approx) have a relative error
    larger than the given ``rel_ftol``.

    """
    results = []
    if isinstance(xs, float) or isinstance(xs, np.ndarray):
        xs = [xs]
    for x in xs:
        if isinstance(x, float):
            results.append(_deriv_error(f, g, x, eps_x, order))
        elif isinstance(x, np.ndarray):
            results.extend(_deriv_error_array(f, g, x, eps_x, order, nrep, weights))
        else:
            raise NotImplementedError
    # make arrays
    deltas = np.array([item[0] for item in results]).ravel()
    deltas_approx = np.array([item[1] for item in results]).ravel()
    # sort
    order = deltas.argsort()
    deltas = deltas[order]
    deltas_approx = deltas_approx[order]
    # chop part of
    ndiscard = int(len(deltas)*discard)
    deltas = deltas[ndiscard:]
    deltas_approx = deltas_approx[ndiscard:]
    # some info on screen
    if verbose:
        print 'Number of comparisons: %5i' % len(deltas)
        with np.errstate(divide='ignore'):
            ratios = (deltas_approx - deltas)/abs(deltas)
        print 'Min relative error:       %10.3e' % np.nanmin(ratios)
        print 'Max relative error:       %10.3e' % np.nanmax(ratios)
        print 'Abs Min relative error:   %10.3e' % np.nanmin(abs(ratios))
        print 'Abs Max relative error:   %10.3e' % np.nanmax(abs(ratios))
        print 'Threshold:                %10.3e' % rel_ftol
        if np.any(np.isnan(ratios)):
            print 'Warning: encountered NaN.'
        print '~~~~~~~i   ~~~~~delta   ~~~~approx   ~~rel.err.'
        for i in xrange(len(deltas)):
            print '%8i   %10.3e   %10.3e   %10.3e' % (
                i, deltas[i], deltas_approx[i],
                (deltas_approx[i] - deltas[i])/abs(deltas[i]))
    # final test
    assert np.all(abs(deltas - deltas_approx) <= rel_ftol*abs(deltas))

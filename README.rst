.. image:: https://travis-ci.org/tovrstra/derivcheck.svg?branch=master
    :target: https://travis-ci.org/tovrstra/derivcheck

Derivcheck provides a robust and very sensitive checker of analytic partial
derivates. It is intended to be used in unit tests of other projects. See
``basic_example.py`` for a basic example.


Installation
============

Derivcheck can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install derivcheck

Alternatively, you can install derivcheck in your home directory:

.. code:: bash

    pip install derivcheck --user


Background and usage
====================

This module implements a function ``assert_deriv`` that uses Ridders' numerical finite
difference scheme to test the implementation of analytic finite differences. Ridders'
method automatically finds the step size (given an initial upper estimate) and the
polynomial order that result in the best approximation. In practice, this means that 14
digits of precision can be reached with 6 to 12 evaluations of the function of interest.

The implementation of Ridders' method is based on the one from the book "Numerical
Recipes" (http://numerical.recipes/), which is in turn a slight rendition of the original
method as proposed by Ridders. (Ridders, C.J.F. 1982, Advances in Engineering Software,
vol. 4, no. 2, pp. 75–76. https://doi.org/10.1016/S0141-1195(82)80057-0)

It is assumed that you have implemented two functions ``f`` and its derivative or gradient
``g``. The function ``f`` takes one argument: a scalar or array with shape ``shape_in``.
It returns a scalar or an array with shape ``shape_out``. The function ``g`` has the same
input but returns a scalar or an array with shape (``shape_out + shape_in``).

The consistency of ``f`` and ``g``, can then be tested around a input value ``origin``
with the following code:

.. code:: python

    assert_deriv(f, g, origin)

where ``origin`` is a scalar or array with shape ``shape_in``, depending on what ``f`` and
``g`` expect as input. An ``AssertionError`` is raised when the gradient function ``g`` is
not consistent with numerical derivatives of ``f``. If Ridders' method does not converge
to sufficiently accurate estimates of a derivative, a ``FloatingPointError`` is raised.

The function ``assert_deriv`` takes several optional arguments to tune its behavior:


* ``widths`` : ``float`` or ``np.ndarray`` (default ``1e-4``)

  The initial (maximal) step size for the finite difference method. Do not take a value
  that is too small. When an array is given, each matrix element of the input of the
  function gets a different step size. When a matrix element is set to zero, the
  derivative towards that element is not test. The function will not be sampled beyond
  [origin-widths, origin+widths].

* ``output_mask`` : ``np.ndarray`` or ``None`` (default)

  This option is useful when the function returns an array outout: it allows the caller to
  select which components of the output need to be tested.

* ``rtol`` : ``float``

  The allowed relative error on the derivative.

* ``atol`` : ``float``

  The allowed absolute error on the derivative.


Release history
===============

- **2017-07-30** 1.0.1

  Fix some missing files and extend README

- **2017-07-28** 1.0.0

  - Ridders' finite difference scheme for testing analytic derivatives.
  - Fully deterministic procedure.
  - More intuitive API

- **2017-07-27** 0.1.0

  Code is made Python 3 compatible and still works with 2.7. Some packaging
  improvements.

- **2017-07-27** 0.0.0

  Initial version: code taken from the Romin project (with contributions and
  ideas from Michael Richer and Paul W. Ayers). Some bugs were fixed through QA
  and CI (pylint, pycodestyle, pydocstyle, nosetests and coverage).


Testing
=======

First you need to get the source package and unzip it, or you can clone the repository. In
the source tree, you simply run:

.. code:: bash

    ./setup.py nosetests

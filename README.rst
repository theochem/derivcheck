.. image:: https://travis-ci.org/tovrstra/derivcheck.svg?branch=master
    :target: https://travis-ci.org/tovrstra/derivcheck

Derivcheck provides a robust and very sensitive checker of analytic partial
derivates. It is intended to be used in unit tests of other projects. See
`basic_example.py` for a basic example.

Installation
============

Derivcheck can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install derivcheck

Alternatively, you can install derivcheck in your home directory:

.. code:: bash

    pip install derivcheck --user


Summary
=======

This module implements a function ``assert_deriv`` that uses Ridders' finite difference
scheme to test the implementation of analytic finite differences. The implementation of
Ridders' method is based on the one from the book "Numerical Recipes"
(http://numerical.recipes/), which is in turn a slight rendition of the method proposed by
Ridders. (Ridders, C.J.F. 1982, Advances in Engineering Software, vol. 4, no. 2, pp.
75–76. https://doi.org/10.1016/S0141-1195(82)80057-0)

It is assumed that you have implemented two functions ``f`` and its derivative or gradient
``g``. The function ``f`` takes one argument: a scalar or array with shape ``shape_in``.
It returns a scalar or an array with shape ``shape_out``. The function ``g`` has the same
input but returns a scalar or an array with shape (``shape_out + shape_in``).

The consistency of ``f`` and ``g``, can then be tested around a input value ``origin``
with the following code:

.. code:: python

    assert_deriv(f, g, origin)

where ``origin`` is a scalar or array with shape ``shape_in``, depending on what ``f``
expects as input. An ``AssertionError`` is raised when the gradient function ``g`` is not
consistent with numerical derivatives of ``f``. If Ridders' method does not converge to
sufficiently accurate estimates of a derivative, a ``FloatingPointError`` is raised.

The function ``assert_deriv`` takes several optional arguments to tune its behavior,
which are documented in the docstring.


Release history
===============

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

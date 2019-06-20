.. image:: https://travis-ci.org/theochem/derivcheck.svg?branch=master
    :target: https://travis-ci.org/theochem/derivcheck
.. image:: https://anaconda.org/theochem/derivcheck/badges/version.svg
    :target: https://anaconda.org/theochem/derivcheck
.. image:: https://codecov.io/gh/theochem/derivcheck/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/theochem/derivcheck

Derivcheck provides a robust and very sensitive checker of analytic partial
derivates. It is intended to be used in unit tests of other projects. See
``deriv_check/basic_example.py`` for a basic example.


Installation
============

Derivcheck can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install derivcheck

Alternatively, you can install derivcheck in your home directory:

.. code:: bash

    pip install derivcheck --user

Lastly, you can also install derivcheck with conda. (See
https://www.continuum.io/downloads)

.. code:: bash

    conda install -c theochem derivcheck


Testing
=======

The tests can be executed as follows:

.. code:: bash

    pytest derivcheck


Background and usage
====================

This module implements a function ``assert_deriv``, which uses Ridders' numerical finite
difference scheme to test the implementation of analytic finite differences. Ridders'
method automatically finds the step size (given an initial upper estimate) and the
polynomial order that result in the best approximation. In practice, this means that 14
digits of precision can be reached with 6 to 12 evaluations of the function of interest.

The implementation of Ridders' method is based on the one from the book "Numerical
Recipes" (http://numerical.recipes/), which is in turn a slight rendition of the original
method as proposed by Ridders. (Ridders, C.J.F. 1982, Advances in Engineering Software,
vol. 4, no. 2, pp. 75–76. https://doi.org/10.1016/S0141-1195(82)80057-0)

It is assumed that you have implemented two functions: ``f`` and its derivative or
gradient ``g``. The function ``f`` takes one argument: a scalar or array with shape
``shape_in``. It returns a scalar or an array with shape ``shape_out``. The function ``g``
has the same input but returns a scalar or an array with shape (``shape_out + shape_in``).

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
  derivative towards that element is not tested. The function will not be sampled beyond
  [origin-widths, origin+widths].

* ``output_mask`` : ``np.ndarray`` or ``None`` (default)

  This option is useful when the function returns an array output: it allows the caller to
  select which components of the output need to be tested. When not given, all components
  are tested.

* ``rtol`` : ``float``

  The allowed relative error on the derivative.

* ``atol`` : ``float``

  The allowed absolute error on the derivative.


Release history
===============

- **2019-06-20** 1.1.5

  Fix a bug related to sharing references to the origin argument of OneDumWrapper.
  Start using Roberto to drive the CI.

- **2017-09-21** 1.1.4

  New template for travis.yml, only affects testing

- **2017-08-22** 1.1.3

  Switch to theochem channel on anaconda.

- **2017-08-01** 1.1.2

  Remove unused dependency on future.

- **2017-08-01** 1.1.1

  Fix dependencies to simplify testing.

- **2017-08-01** 1.1.0

  - Tests are now included with the installed module.
  - Experimental: deployment to github, pypi and anaconda.

- **2017-07-30** 1.0.2

  Updated README and install recipe for Conda

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


How to make a release (Github, PyPI and anaconda.org)
=====================================================

Before you do this, make sure everything is OK. The PyPI releases cannot be undone. If you
delete a file from PyPI (because of a mistake), you cannot upload the fixed file with the
same filename! See https://github.com/pypa/packaging-problems/issues/74

1. Update the release history.
2. Commit the final changes to master and push to github.
3. Wait for the CI tests to pass. Check if the README looks ok, etc. If needed, fix things
   and repeat step 2.
4. Make a git version tag: ``git tag <some_new_version>`` Follow the semantic versioning
   guidelines: http://semver.org
5. Push the tag to github: ``git push origin master --tags``

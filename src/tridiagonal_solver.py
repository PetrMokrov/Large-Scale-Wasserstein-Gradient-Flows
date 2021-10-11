######################################
# !CREDITS!
# The code is taken from `pychangcooper` project:
# link: https://github.com/grburgess/pychangcooper

import numpy as np
from numba import jit, njit


@njit(fastmath=True)
def jit_forward_sweep(n, cprime, dprime, a, b, c, d):
    for i in range(1, n):
        b_minus_ac = b[i] - a[i] * cprime[i - 1]

        cprime[i] = c[i] / b_minus_ac

        dprime[i] = (d[i] - a[i] * dprime[i - 1]) / b_minus_ac


@njit(fastmath=True)
def jit_backward_sweep(n, n_j_plus_1, cprime, dprime):
    for j in range(n - 2, -1, -1):
        n_j_plus_1[j] = dprime[j] - cprime[j] * n_j_plus_1[j + 1]


class TridiagonalSolver(object):
    def __init__(self, a, b, c):
        """
        A tridiagonal solver for the equation:

        a_i x_i-1 + b_i x_i _ c_i x_i+1 = d_i

        :param a: the x_i-1 terms
        :param b: the x_i terms
        :param c: the x_i+1 terms


        """

        # get the number of elements
        self._n_grid_points = len(a)

        # make sure the elements are all the same
        # size
        assert len(a) == len(b)
        assert len(a) == len(c)

        # assign the elements to the class
        self._a = np.array(a)
        self._b = np.array(b)
        self._c = np.array(c)

        # if the a terms are all zero, we do not need the
        # forward sweep as a is already eliminated

        self._a_non_zero = ~np.all(self._a == 0)

        # the 0th term of the c prime will always
        # be the same so pre compute.
        # is a = 0 then all terms will be the same.

        # this is basically the first step of the forward
        # sweep

        self._cprime = self._c / self._b

    def _forward_sweep(self, d):
        """
        This is the forward sweep of the tridiagonal solver


        """

        # we have already set the first terms in c prime
        # if we need to forward sweep, we must set the remaining
        # terms. Otherwise, they are just ratios

        self._cprime = self._c / self._b
        self._dprime = d / self._b

        if self._a_non_zero:
            jit_forward_sweep(
                self._n_grid_points,
                self._cprime,
                self._dprime,
                self._a,
                self._b,
                self._c,
                d,
            )

    def _backwards_substitution(self):
        """
        This is the backwards substitution step of the tridiagonal
        solver.
        """

        n_j_plus_1 = np.zeros(self._n_grid_points)

        # set the end points
        n_j_plus_1[-1] = self._dprime[-1]

        # backwards step to the beginning

        jit_backward_sweep(self._n_grid_points, n_j_plus_1, self._cprime, self._dprime)

        return n_j_plus_1

    def solve(self, d_j):
        """"""

        self._forward_sweep(d_j)

        d_j_plus_one = self._backwards_substitution()

        return d_j_plus_one

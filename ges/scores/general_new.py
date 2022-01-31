# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Module containing the DecomposableScore class, inherited by all
classes which implement a locally decomposable score for directed
acyclic graphs. By default, the class also caches the results of
computing local scores.

NOTE: It is not mandatory to inherit this class when developing custom
scores to use with the GES implementation in ges.py. The only
requirement is that the class defines:
  1. the local_score function (see below),
  2. an attribute "p" for the total number of variables.

"""

import numpy as np
from .decomposable_score import DecomposableScore
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------
# l0-penalized general score for a sample from a single
# (observational) environment


class GeneralScore(DecomposableScore):
    """
    Implements a cached l0-penalized general score.

    """

    def __init__(self, data, clip=1, cache=False, debug=0):
        super().__init__(data, cache=cache, debug=debug)

        self.n, self.p = data.shape
        self.data = data
        self.sensitivity = self.clip = clip

    def _linear_regression(self, X, y, epochs=100, lr=1e-1):
        _, reg_size = X.shape
        theta = np.zeros(reg_size)
        bias = 0
        for i in range(epochs):
            y_curr = X @ theta + bias
            thrsh_errs = np.clip(y - y_curr, - np.sqrt(self.clip), np.sqrt(self.clip))
            #print(X.shape, thrsh_errs.shape)
            cost = sum([err**2 for err in thrsh_errs])
            theta_grad = - (2/self.n) * sum(X.T @ thrsh_errs)
            bias_grad = - (2/self.n) * sum(thrsh_errs)
            theta -= lr * theta_grad
            bias -= lr * bias_grad
        return theta, bias, cost

    def _compute_local_score(self, x, pa):
        """
        Compute the local score of a given node and a set of
        parents.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the corresponding score

        """
        l0_term = (np.log(self.n) / self.n) * (len(pa) + 1)
        y = self.data[:, x]
        if len(pa) > 0:
            X = self.data[:, list(pa)]
            #print(X.shape)
            _, _, mean_rss = self._linear_regression(X, y)
        else:
            mean_rss = sum(np.minimum(y**2, self.clip))
        local_score = - mean_rss - l0_term
        return local_score

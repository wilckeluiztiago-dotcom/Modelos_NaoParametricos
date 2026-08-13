"""
Estimador de Nadaraya-Watson para a função de deriva.
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from .kernels import kernel_gaussiano


def nadaraya_watson_deriva(precos, h=None, grid=None, n_grid=150):
    """
    Estimador local da deriva a partir de uma série de preços discretos.

    Implementa a versão discretizada do estimador contínuo
    do Capítulo 9:

        b̂_h(x) = Σ K_h(X_k - x) ΔX_k  /  Σ K_h(X_k - x) Δt

    Parameters
    ----------
    precos : array-like ou Series
        Série de preços (ou de qualquer processo de difusão observado).
    h : float, optional
        Largura de banda. Se None, usa regra de Silverman.
    grid : array-like, optional
        Pontos onde a deriva será avaliada.
    n_grid : int
        Número de pontos da grade quando grid=None.

    Returns
    -------
    grid : ndarray
    b_hat : ndarray
    h : float
        Largura de banda efetivamente utilizada.
    """
    X = np.asarray(precos, dtype=float).ravel()
    dX = np.diff(X)
    X_lag = X[:-1]
    n = len(X_lag)

    if h is None:
        h = 1.06 * np.std(X_lag) * n ** (-0.2)

    if grid is None:
        grid = np.linspace(X.min() * 0.95, X.max() * 1.05, n_grid)
    else:
        grid = np.asarray(grid, dtype=float)

    b_hat = np.zeros_like(grid)
    for i, x in enumerate(grid):
        u = (X_lag - x) / h
        K = kernel_gaussiano(u)
        den = np.sum(K)
        if den > 1e-10:
            b_hat[i] = np.sum(K * dX) / den
        else:
            b_hat[i] = 0.0

    return grid, b_hat, h

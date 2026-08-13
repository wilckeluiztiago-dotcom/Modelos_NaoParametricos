"""
Kernels e estimadores de densidade.
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import gamma


def kernel_gaussiano(u):
    """Kernel gaussiano padrão (segunda ordem)."""
    return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)


def estimador_densidade_kernel(dados, grid, h):
    """
    Estimador de Rosenblatt-Parzen com kernel gaussiano.

    f̂_h(x) = (1/(n h)) Σ K((x - X_i)/h)
    """
    dados = np.asarray(dados, dtype=float)
    f = np.zeros_like(grid, dtype=float)
    for i, x in enumerate(grid):
        f[i] = np.mean(kernel_gaussiano((dados - x) / h)) / h
    return f


def estimador_densidade_gama_chen(dados, grid, h):
    """
    Estimador de densidade por kernel gama de Chen
    (suporte [0, ∞), livre de viés de fronteira em zero).

    K_{G(x/h+1, h)}(y) = y^{x/h} exp(-y/h) / [h^{x/h+1} Γ(x/h+1)]
    """
    dados = np.asarray(dados, dtype=float)
    f = np.zeros_like(grid, dtype=float)
    for i, x in enumerate(grid):
        if x <= 0:
            f[i] = 0.0
            continue
        shape = x / h + 1.0
        scale = h
        f[i] = np.mean(gamma.pdf(dados, a=shape, scale=scale))
    return f

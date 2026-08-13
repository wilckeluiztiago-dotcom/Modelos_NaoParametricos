"""
Simulação de EDEs e calibração do modelo de Vasicek.
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def simular_euler_maruyama(b_func, sigma_func, x0, T, n_steps, n_paths=1000, seed=None):
    """
    Gera n_paths trajetórias pelo esquema de Euler-Maruyama.

        Y_{k+1} = Y_k + b(Y_k) Δt + σ(Y_k) √Δt Z_{k+1}

    Parameters
    ----------
    b_func, sigma_func : callable
        Funções de deriva e difusão (aceitam array).
    x0 : float
        Condição inicial.
    T : float
        Horizonte temporal.
    n_steps : int
        Número de passos de discretização.
    n_paths : int
        Número de trajetórias independentes.
    seed : int, optional
        Semente para reprodutibilidade.

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    for k in range(n_steps):
        Z = np.random.normal(0.0, 1.0, size=n_paths)
        x = paths[:, k]
        paths[:, k + 1] = x + b_func(x) * dt + sigma_func(x) * np.sqrt(dt) * Z

    return paths


def vasicek_parametros(r):
    """
    Calibração clássica do modelo de Vasicek por regressão linear
    dos incrementos:

        Δr = a + b r + ε  →  κ = -b,  θ = -a/b,  σ = std(ε)
    """
    r = np.asarray(r, dtype=float).ravel()
    dr = np.diff(r)
    r_lag = r[:-1]

    A = np.column_stack([np.ones_like(r_lag), r_lag])
    beta, residuals, _, _ = np.linalg.lstsq(A, dr, rcond=None)
    a, b = beta
    kappa = -b
    theta = -a / b if abs(b) > 1e-12 else np.mean(r)
    resid = dr - (a + b * r_lag)
    sigma = np.std(resid)
    return float(kappa), float(theta), float(sigma)

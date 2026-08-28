"""
Módulo 17 – Processos Gaussianos e Inferência Bayesiana Não-Paramétrica
Inspirado em: Capítulo 32 – Processos Gaussianos, kernels Matérn, kriging
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def kernel_rbf(x1: np.ndarray, x2: np.ndarray, comprimento: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Kernel RBF / Gaussiano (Cap. 32.2)."""
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    dist2 = (x1 - x2.T)**2
    return sigma**2 * np.exp(-0.5 * dist2 / comprimento**2)


def predicao_gp(
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_teste: np.ndarray,
    comprimento: float = 1.0,
    ruido: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predição por Processo Gaussiano (Kriging – Cap. 32.5).
    """
    K = kernel_rbf(x_treino, x_treino, comprimento) + ruido * np.eye(len(x_treino))
    K_s = kernel_rbf(x_treino, x_teste, comprimento)
    K_ss = kernel_rbf(x_teste, x_teste, comprimento)

    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_treino))
        media = K_s.T @ alpha
        v = np.linalg.solve(L, K_s)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
    except np.linalg.LinAlgError:
        media = np.zeros(len(x_teste))
        var = np.ones(len(x_teste))

    return media, np.maximum(var, 0)


if __name__ == "__main__":
    print("[17] Processos Gaussianos carregados.")

"""
Módulo 18 – Espaços de Hilbert Reprodutores (RKHS) e Métodos de Kernel
Inspirado em: Capítulo 34 – Teorema de Moore-Aronszajn e Representador
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def kernel_polinomial(x1: np.ndarray, x2: np.ndarray, grau: int = 2) -> np.ndarray:
    return (1 + x1.reshape(-1, 1) @ x2.reshape(1, -1))**grau


def representador_ridge(
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_teste: np.ndarray,
    lambda_reg: float = 0.1,
    kernel_func=None
) -> np.ndarray:
    """
    Teorema do Representador + ridge regression em RKHS (Cap. 34.3).
    """
    if kernel_func is None:
        from scipy.stats import norm
def kernel_rbf(x1, x2, comprimento=1.0, sigma=1.0):
    x1=x1.reshape(-1,1); x2=x2.reshape(-1,1)
    return sigma**2 * np.exp(-0.5*((x1-x2.T)**2)/comprimento**2)
        kernel_func = lambda a, b: kernel_rbf(a, b, 1.0)

    K = kernel_func(x_treino, x_treino)
    n = len(x_treino)
    alpha = np.linalg.solve(K + lambda_reg * np.eye(n), y_treino)
    K_teste = kernel_func(x_teste, x_treino)
    return K_teste @ alpha


if __name__ == "__main__":
    print("[18] RKHS carregado.")

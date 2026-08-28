"""
Módulo 04 – Regressão Não-Paramétrica de Nadaraya-Watson
Inspirado em: Capítulo 2.2 e Capítulo 9 (versão contínua para difusões)
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm

def kernel_gaussiano(u):
    return norm.pdf(u)

def kernel_epanechnikov(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def nadaraya_watson(
    x_avaliacao: np.ndarray,
    x_dados: np.ndarray,
    y_dados: np.ndarray,
    h: float,
    kernel: str = "gaussiano"
) -> np.ndarray:
    """
    Estimador de Nadaraya-Watson:

    ˆm_h(x) = Σ K((x-X_i)/h) Y_i  /  Σ K((x-X_i)/h)

    Usado para suavizar a curva de luz e revelar a forma do trânsito.
    """
    x_avaliacao = np.asarray(x_avaliacao).ravel()
    x_dados = np.asarray(x_dados).ravel()
    y_dados = np.asarray(y_dados).ravel()

    if kernel == "gaussiano":
        K = kernel_gaussiano
    else:
        K = kernel_epanechnikov

    m_hat = np.zeros_like(x_avaliacao, dtype=float)

    for i, x in enumerate(x_avaliacao):
        pesos = K((x - x_dados) / h)
        soma_pesos = np.sum(pesos)
        if soma_pesos > 1e-12:
            m_hat[i] = np.sum(pesos * y_dados) / soma_pesos
        else:
            m_hat[i] = np.nan

    return m_hat


def nadaraya_watson_vetorizado(
    x_avaliacao: np.ndarray,
    x_dados: np.ndarray,
    y_dados: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Versão vetorizada (mais rápida) usando broadcasting.
    """
    x_avaliacao = np.asarray(x_avaliacao).ravel()
    x_dados = np.asarray(x_dados).ravel()
    y_dados = np.asarray(y_dados).ravel()

    # Matriz de diferenças (n_eval, n_dados)
    u = (x_avaliacao[:, None] - x_dados[None, :]) / h
    pesos = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)  # Gaussiano

    soma_pesos = pesos.sum(axis=1)
    soma_pesos = np.where(soma_pesos < 1e-12, 1e-12, soma_pesos)
    m_hat = (pesos * y_dados[None, :]).sum(axis=1) / soma_pesos
    return m_hat


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + rng.normal(0, 0.2, 200)
    suavizado = nadaraya_watson_vetorizado(x, x, y, h=0.4)
    print(f"[04] Nadaraya-Watson – amplitude suavizada: {np.ptp(suavizado):.3f}")

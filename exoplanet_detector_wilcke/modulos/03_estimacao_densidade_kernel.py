"""
Módulo 03 – Estimação de Densidade por Kernel (Rosenblatt-Parzen)
Inspirado em: Capítulo 2 – O Estimador de Rosenblatt-Parzen para Densidades
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm


def kernel_gaussiano(u: np.ndarray) -> np.ndarray:
    """Kernel Gaussiano padrão K(u) = (1/√(2π)) exp(-u²/2)."""
    return norm.pdf(u)


def kernel_epanechnikov(u: np.ndarray) -> np.ndarray:
    """Kernel de Epanechnikov (ótimo AMISE)."""
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def estimador_rosenblatt_parzen(
    pontos: np.ndarray,
    dados: np.ndarray,
    h: float,
    kernel: str = "gaussiano"
) -> np.ndarray:
    """
    Estimador de densidade de Rosenblatt-Parzen.

    ˆf_{X,h}(x) = (1/(n h)) Σ K((x - X_i)/h)

    Parâmetros
    ----------
    pontos : array onde avaliar a densidade
    dados  : amostra (fluxo ou tempo)
    h      : largura de banda
    kernel : 'gaussiano' ou 'epanechnikov'
    """
    dados = np.asarray(dados).ravel()
    pontos = np.asarray(pontos).ravel()
    n = len(dados)

    if kernel == "gaussiano":
        K = kernel_gaussiano
    elif kernel == "epanechnikov":
        K = kernel_epanechnikov
    else:
        raise ValueError("Kernel não suportado")

    densidade = np.zeros_like(pontos, dtype=float)
    for i, x in enumerate(pontos):
        u = (x - dados) / h
        densidade[i] = np.mean(K(u)) / h

    return densidade


def mise_assintotico(h: float, n: int, R_K: float = 0.28209479, mu2: float = 1.0) -> float:
    """
    Aproximação do MISE assintótico (Cap. 2.1.1).
    R(K) ≈ 1/(2√π) para Gaussiano.
    """
    # Termo de variância + viés² (requer f'' desconhecido – usa heurística)
    termo_var = R_K / (n * h)
    # Para demonstração, assume f'' típico de ordem 1
    termo_vies = (mu2**2 / 4) * h**4 * 1.0
    return termo_var + termo_vies


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    amostra = rng.normal(0, 1, 1000)
    x_grid = np.linspace(-3, 3, 200)
    dens = estimador_rosenblatt_parzen(x_grid, amostra, h=0.3)
    print(f"[03] Densidade estimada – máximo: {dens.max():.4f}")

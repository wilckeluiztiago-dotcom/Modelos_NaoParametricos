"""
Módulo 11 – Kernels Assimétricos para Estimação em Fronteira
Inspirado em: Capítulos 12 e 13 – Birnbaum-Saunders, Log-Normal, Gaussiano Inverso
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import invgauss, lognorm


def kernel_birnbaum_saunders(u: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Kernel baseado na densidade Birnbaum-Saunders (Cap. 12.2).
    Útil para dados positivos (fluxos residuals após normalização).
    """
    u = np.asarray(u)
    # Forma simplificada
    termo = (u - 1/u) / alpha
    dens = (1 / (alpha * u * np.sqrt(2 * np.pi))) * np.exp(-0.5 * termo**2) * (1 + 1/u**2) / 2
    return np.where(u > 0, dens, 0.0)


def kernel_lognormal(u: np.ndarray, s: float = 0.5) -> np.ndarray:
    """Kernel Log-Normal (Cap. 12.3)."""
    u = np.asarray(u)
    return np.where(u > 0, lognorm.pdf(u, s), 0.0)


def kernel_gaussiano_inverso(u: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """Kernel Gaussiano Inverso (Cap. 13)."""
    u = np.asarray(u)
    return np.where(u > 0, invgauss.pdf(u, mu), 0.0)


def estimador_fronteira_assimetrico(
    pontos: np.ndarray,
    dados: np.ndarray,
    h: float,
    tipo: str = "gama"
) -> np.ndarray:
    """
    Estimador de densidade com kernel assimétrico para suporte [0,∞).
    """
    dados = np.asarray(dados).ravel()
    pontos = np.asarray(pontos).ravel()
    dens = np.zeros_like(pontos)

    for i, x in enumerate(pontos):
        if tipo == "lognormal":
            u = (dados) / (x + h)  # reescalonamento típico
            dens[i] = np.mean(kernel_lognormal(u)) / h
        else:
            # Fallback Gama simples
            dens[i] = np.mean(np.exp(-dados / h) * (dados / h) / h)  # heurística

    return dens


if __name__ == "__main__":
    print("[11] Kernels assimétricos carregados.")

"""
Módulo 16 – Estimador da Função de Vida Média Residual (MRL)
Inspirado em: Capítulo 21 – Vida Média Residual suavizada
Autor: Luiz Tiago Wilcke
"""

import numpy as np


def vida_media_residual(
    t: float,
    dados: np.ndarray,
    h: float = 0.1
) -> float:
    """
    Estimador suave da MRL:

    m(t) = E[X - t | X > t]

    Aplicado a tempos de trânsito ou profundidades residuals.
    """
    dados = np.asarray(dados).ravel()
    acima = dados[dados > t]
    if len(acima) < 5:
        return np.nan
    return float(np.mean(acima) - t)


def mrl_suavizada(
    grid_t: np.ndarray,
    dados: np.ndarray
) -> np.ndarray:
    """Versão vetorizada."""
    return np.array([vida_media_residual(t, dados) for t in grid_t])


if __name__ == "__main__":
    print("[16] MRL carregado.")
